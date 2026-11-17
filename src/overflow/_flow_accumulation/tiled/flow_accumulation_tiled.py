import concurrent.futures
import math
import queue
import time
from threading import Lock

import numba
import numpy as np
from numba import njit  # type: ignore[attr-defined]
from osgeo import gdal

from overflow._flow_accumulation.core.flow_accumulation import (
    _parse_weights_nodata_mode,
    get_next_cell,
    single_tile_flow_accumulation,
    single_tile_flow_accumulation_weighted,
)
from overflow._util.constants import (
    FLOW_ACCUMULATION_NODATA,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    WEIGHTED_ACCUMULATION_NODATA,
)
from overflow._util.perimeter import get_tile_perimeter
from overflow._util.progress import ProgressCallback, ProgressTracker, silent_callback
from overflow._util.raster import (
    RasterChunk,
    create_dataset,
    open_dataset,
    raster_chunker,
)

from .global_state import GlobalState, WeightedGlobalState


@njit(nogil=True)
def calculate_flow_accumulation_tile(
    flow_direction: np.ndarray, tile_row: int, tile_col: int
):
    """
    Calculate flow accumulation for a single tile.

    This function processes a single tile of the flow direction raster to compute
    flow accumulation and extract perimeter information needed for global processing.

    Args:
        flow_direction (np.ndarray): Flow direction data for the tile.
        tile_row (int): Row index of the tile in the grid.
        tile_col (int): Column index of the tile in the grid.

    Returns:
        tuple: Contains flow accumulation data and perimeter information.
    """
    # Compute flow accumulation and links for the tile
    flow_accumulation, links = single_tile_flow_accumulation(
        flow_direction, True, tile_row, tile_col
    )

    # Extract perimeter information for global processing
    flow_acc_perimeter = get_tile_perimeter(flow_accumulation)
    flow_dir_perimeter = get_tile_perimeter(flow_direction)
    links_row_perimeter = get_tile_perimeter(links[:, :, 0])
    links_col_perimeter = get_tile_perimeter(links[:, :, 1])

    return (
        flow_accumulation,
        flow_acc_perimeter,
        flow_dir_perimeter,
        links_row_perimeter,
        links_col_perimeter,
        tile_row,
        tile_col,
    )


@njit(nogil=True)
def _walk_offset_downstream(
    flow_acc: np.ndarray,
    flow_dir: np.ndarray,
    tile_row: int,
    tile_col: int,
    start_row: int,
    start_col: int,
    value,
    global_state,
    stop_map: dict,
    stop_at_nodes: bool,
):
    """Add value to each cell downstream of (start_row, start_col) within the tile."""
    rows, cols = flow_acc.shape
    current_row, current_col = start_row, start_col
    # sentinel must not collide with out-of-tile coordinates
    prev_row, prev_col = -9999, -9999
    current_dir = flow_dir[current_row, current_col]
    # Cycle detection: no valid path should exceed total cells in tile
    max_iterations = rows * cols
    iterations = 0
    while current_dir not in (FLOW_DIRECTION_NODATA, FLOW_DIRECTION_UNDEFINED):
        iterations += 1
        if iterations > max_iterations:
            print(
                "Warning: Cycle detected in flow direction data during finalization. tile:",
                tile_row,
                tile_col,
                "cell:",
                current_row,
                current_col,
            )
            break
        # Bounds check: ensure we're still within the tile
        if not (0 <= current_row < rows and 0 <= current_col < cols):
            break
        flow_acc[current_row, current_col] += value
        next_row, next_col, next_dir = get_next_cell(
            flow_dir, current_row, current_col
        )
        if next_row == prev_row and next_col == prev_col:
            # Two-cell cycle: stop before applying the value repeatedly
            print(
                "Warning: Cycle detected in flow direction data during finalization. tile:",
                tile_row,
                tile_col,
                "cell:",
                current_row,
                current_col,
            )
            break
        prev_row, prev_col = current_row, current_col
        current_row, current_col, current_dir = next_row, next_col, next_dir
        next_global_index = global_state.get_global_cell_index(
            tile_row, tile_col, current_row, current_col
        )
        if (
            stop_at_nodes
            and next_global_index in stop_map
            and stop_map[next_global_index]
        ):
            # Flow reaches a cell accounted for in global accumulation
            break


@njit(nogil=True)
def finalize_flow_accumulation(
    flow_acc: np.ndarray,
    flow_dir: np.ndarray,
    global_acc: dict,
    global_offset: dict,
    tile_row: int,
    tile_col: int,
    global_state: GlobalState,
    source_index: np.ndarray,
    source_value: np.ndarray,
):
    """
    Finalize flow accumulation for a tile using global accumulation data.

    Graph-node offsets are walked downstream until the next cell with a computed
    global accumulation, whose offset already contains the inflow. Upstream
    sources are walked to the tile exit, since their values are not carried by
    the graph.

    Args:
        flow_acc (np.ndarray): Initial flow accumulation for the tile.
        flow_dir (np.ndarray): Flow direction data for the tile.
        global_acc (dict): Global accumulation values.
        global_offset (dict): Global offset values.
        tile_row (int): Row index of the tile.
        tile_col (int): Column index of the tile.
        global_state (GlobalState): Global state object containing grid information.
        source_index (np.ndarray): Global indices of upstream source cells.
        source_value (np.ndarray): Upstream source values, aligned with source_index.

    Returns:
        tuple: Updated flow accumulation array and tile indices.
    """
    rows, cols = flow_acc.shape
    chunk_size = global_state.chunk_size

    for global_index in global_offset:
        global_row = global_index // (global_state.num_cols * chunk_size)
        global_col = global_index % (global_state.num_cols * chunk_size)
        local_row = global_row - tile_row * chunk_size
        local_col = global_col - tile_col * chunk_size
        if 0 <= local_row < rows and 0 <= local_col < cols:
            _walk_offset_downstream(
                flow_acc,
                flow_dir,
                tile_row,
                tile_col,
                local_row,
                local_col,
                global_offset[global_index],
                global_state,
                global_acc,
                True,
            )

    for k in range(source_index.shape[0]):
        global_index = source_index[k]
        global_row = global_index // (global_state.num_cols * chunk_size)
        global_col = global_index % (global_state.num_cols * chunk_size)
        local_row = global_row - tile_row * chunk_size
        local_col = global_col - tile_col * chunk_size
        if 0 <= local_row < rows and 0 <= local_col < cols:
            _walk_offset_downstream(
                flow_acc,
                flow_dir,
                tile_row,
                tile_col,
                local_row,
                local_col,
                source_value[k],
                global_state,
                global_acc,
                False,
            )

    return flow_acc, tile_row, tile_col


@njit(nogil=True)
def find_tile_exit(flow_dir: np.ndarray, row: int, col: int):
    """
    Follow flow from a cell within a tile until it leaves the tile.

    Args:
        flow_dir (np.ndarray): Flow direction data for the tile.
        row (int): Starting row within the tile.
        col (int): Starting column within the tile.

    Returns:
        tuple: (exit_row, exit_col, exited). When exited is True, (exit_row,
            exit_col) is the out-of-tile cell the flow enters next. When False,
            the flow terminates inside the tile (or hits a cycle) and the
            exit coordinates are meaningless.
    """
    rows, cols = flow_dir.shape
    current_row, current_col = row, col
    current_dir = flow_dir[current_row, current_col]
    max_iterations = rows * cols
    iterations = 0
    while current_dir not in (FLOW_DIRECTION_NODATA, FLOW_DIRECTION_UNDEFINED):
        iterations += 1
        if iterations > max_iterations:
            return 0, 0, False
        next_row, next_col, next_dir = get_next_cell(
            flow_dir, current_row, current_col
        )
        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
            return next_row, next_col, True
        current_row, current_col, current_dir = next_row, next_col, next_dir
    return 0, 0, False


@njit(nogil=True)
def calculate_flow_accumulation_tile_weighted(
    flow_direction: np.ndarray,
    weights: np.ndarray,
    weights_nodata: float,
    nodata_mode: int,
    tile_row: int,
    tile_col: int,
):
    """
    Calculate weighted flow accumulation for a single tile.

    Weighted analog of `calculate_flow_accumulation_tile`: sums the weights
    raster over each cell's upstream contributing area within the tile,
    instead of counting cells.

    Args:
        flow_direction (np.ndarray): Flow direction data for the tile.
        weights (np.ndarray): Weights data for the tile, same shape as flow_direction.
        weights_nodata (float): The nodata value used in the weights raster.
        nodata_mode (int): WEIGHTS_NODATA_MODE_ZERO or WEIGHTS_NODATA_MODE_PROPAGATE.
        tile_row (int): Row index of the tile in the grid.
        tile_col (int): Column index of the tile in the grid.

    Returns:
        tuple: Contains flow accumulation data and perimeter information.
    """
    flow_accumulation, links = single_tile_flow_accumulation_weighted(
        flow_direction,
        weights,
        weights_nodata,
        nodata_mode,
        True,
        tile_row,
        tile_col,
    )

    flow_acc_perimeter = get_tile_perimeter(flow_accumulation)
    flow_dir_perimeter = get_tile_perimeter(flow_direction)
    links_row_perimeter = get_tile_perimeter(links[:, :, 0])
    links_col_perimeter = get_tile_perimeter(links[:, :, 1])

    return (
        flow_accumulation,
        flow_acc_perimeter,
        flow_dir_perimeter,
        links_row_perimeter,
        links_col_perimeter,
        tile_row,
        tile_col,
    )


@njit(nogil=True)
def finalize_flow_accumulation_weighted(
    flow_acc: np.ndarray,
    flow_dir: np.ndarray,
    global_acc: dict,
    global_offset: dict,
    global_acc_is_external: dict,
    tile_row: int,
    tile_col: int,
    global_state: WeightedGlobalState,
    source_index: np.ndarray,
    source_value: np.ndarray,
):
    """
    Finalize weighted flow accumulation for a tile using global accumulation data.

    Same as `finalize_flow_accumulation`, except walks stop on cells flagged in
    `global_acc_is_external` instead of `global_acc[...] != 0`: in weighted
    "zero" nodata mode a legitimate all-zero-weight upstream area can produce
    a real 0.0 accumulation, which the unweighted `!= 0` check would
    misinterpret as "not yet computed".

    Args:
        flow_acc (np.ndarray): Initial weighted flow accumulation for the tile.
        flow_dir (np.ndarray): Flow direction data for the tile.
        global_acc (dict): Global accumulation values.
        global_offset (dict): Global offset values.
        global_acc_is_external (dict): Whether each global index's accumulation
            was computed from an externally-draining perimeter cell.
        tile_row (int): Row index of the tile.
        tile_col (int): Column index of the tile.
        global_state (WeightedGlobalState): Global state object containing grid information.
        source_index (np.ndarray): Global indices of upstream source cells.
        source_value (np.ndarray): Upstream source values, aligned with source_index.

    Returns:
        tuple: Updated flow accumulation array and tile indices.
    """
    rows, cols = flow_acc.shape
    chunk_size = global_state.chunk_size

    for global_index in global_offset:
        global_row = global_index // (global_state.num_cols * chunk_size)
        global_col = global_index % (global_state.num_cols * chunk_size)
        local_row = global_row - tile_row * chunk_size
        local_col = global_col - tile_col * chunk_size
        if 0 <= local_row < rows and 0 <= local_col < cols:
            _walk_offset_downstream(
                flow_acc,
                flow_dir,
                tile_row,
                tile_col,
                local_row,
                local_col,
                global_offset[global_index],
                global_state,
                global_acc_is_external,
                True,
            )

    for k in range(source_index.shape[0]):
        global_index = source_index[k]
        global_row = global_index // (global_state.num_cols * chunk_size)
        global_col = global_index % (global_state.num_cols * chunk_size)
        local_row = global_row - tile_row * chunk_size
        local_col = global_col - tile_col * chunk_size
        if 0 <= local_row < rows and 0 <= local_col < cols:
            _walk_offset_downstream(
                flow_acc,
                flow_dir,
                tile_row,
                tile_col,
                local_row,
                local_col,
                source_value[k],
                global_state,
                global_acc_is_external,
                False,
            )

    return flow_acc, tile_row, tile_col


def _compute_upstream_injections(
    upstream_sources: dict[tuple[int, int], int] | None,
    input_band,
    chunk_size: int,
    global_state,
    value_dtype,
):
    """
    Locate where upstream-source inflow enters the global graph.

    Each source adds its value to the flow accumulation of every cell downstream
    of the source. Within the source's own tile this is applied by the finalize
    walk from the source cell (`source_index`/`source_value`). Once the flow
    leaves that tile, the value must travel through the global graph like any
    other inflow, so it is seeded at the entry cell of the neighbouring tile
    (`exit_offsets`) before the graph is solved.

    Args:
        upstream_sources (dict | None): (row, col) -> value.
        input_band: Flow direction raster band, used to read tile flow directions.
        chunk_size (int): Tile size in pixels.
        global_state: GlobalState or WeightedGlobalState for the grid.
        value_dtype: numpy dtype of the accumulation values.

    Returns:
        tuple: (source_index, source_value, exit_offsets, inject_index, inject_value).
            source_index/source_value are the walk starts applied per tile.
            exit_offsets is a dict of entry global index -> value. inject_index/
            inject_value are the same entries as arrays to seed the graph solve.
    """
    source_offsets: dict[int, float] = {}
    exit_offsets: dict[int, float] = {}
    tile_cache: dict[tuple[int, int], np.ndarray] = {}

    if upstream_sources:
        for (row, col), value in upstream_sources.items():
            tile_row = row // chunk_size
            tile_col = col // chunk_size
            local_row = row % chunk_size
            local_col = col % chunk_size
            global_index = global_state.get_global_cell_index(
                tile_row, tile_col, local_row, local_col
            )
            if global_index == -1:
                continue
            source_offsets[global_index] = source_offsets.get(global_index, 0) + value

            key = (tile_row, tile_col)
            if key not in tile_cache:
                flow_dir_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
                flow_dir_tile.read(input_band)
                tile_cache[key] = flow_dir_tile.data
            exit_row, exit_col, exited = find_tile_exit(
                tile_cache[key], local_row, local_col
            )
            if not exited:
                continue

            next_tile_row, next_tile_col = global_state.get_neighboring_tile(
                tile_row, tile_col, exit_row, exit_col
            )
            next_local_row, next_local_col = global_state.adjust_coordinates(
                exit_row, exit_col
            )
            next_global_index = global_state.get_global_cell_index(
                next_tile_row, next_tile_col, next_local_row, next_local_col
            )
            if next_global_index == -1:
                continue
            exit_offsets[next_global_index] = (
                exit_offsets.get(next_global_index, 0) + value
            )

    source_index = np.array(list(source_offsets.keys()), dtype=np.int64)
    source_value = np.array(list(source_offsets.values()), dtype=value_dtype)
    inject_index = np.array(list(exit_offsets.keys()), dtype=np.int64)
    inject_value = np.array(list(exit_offsets.values()), dtype=value_dtype)
    return source_index, source_value, exit_offsets, inject_index, inject_value


def _apply_upstream_sources(global_offset, exit_offsets) -> None:
    """
    Add upstream inflow for entry cells that are not part of the solved graph.

    Entry cells already in the graph received their inflow during the solve.
    Any other entry cell is added here so its finalize walk applies the inflow.
    """
    for index, value in exit_offsets.items():
        if index not in global_offset:
            global_offset[index] = value


def _flow_accumulation_tiled(
    input_path: str,
    output_path: str,
    chunk_size: int,
    progress_callback: ProgressCallback | None = None,
    upstream_sources: dict[tuple[int, int], int] | None = None,
) -> None:
    """
    Compute flow accumulation using a tiled approach for large rasters.

    This function orchestrates the tiled flow accumulation process, including:
    1. Setting up the working environment
    2. Processing individual tiles
    3. Calculating global accumulation
    4. Finalizing tile accumulations based on global data

    Args:
        input_path (str): Path to the input flow direction raster.
        output_path (str): Path for the output flow accumulation raster.
        chunk_size (int): Size of each tile (chunk) in pixels.
        progress_callback (ProgressCallback | None): Optional callback for progress updates.
            If None, the operation runs silently.

    Returns:
        None
    """
    # Setup progress tracking
    if progress_callback is None:
        progress_callback = silent_callback
    tracker = ProgressTracker(progress_callback, "Flow Accumulation", total_steps=3)

    # Setup datasets
    flow_dir_ds, output_ds, no_data_value = setup_datasets(input_path, output_path)
    global_state, input_band, output_band = init_global_state(
        flow_dir_ds, output_ds, no_data_value, chunk_size
    )

    # Setup for parallel processing
    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()

    chunk_counter = [0]  # Use list for mutability in closure
    total_chunks = math.ceil(input_band.YSize / chunk_size) * math.ceil(
        input_band.XSize / chunk_size
    )

    def handle_flow_acc_tile_result(future):
        """Handle the result of a single tile flow accumulation calculation."""
        (
            flow_acc,
            flow_acc_perimeter,
            flow_dir_perimeter,
            links_row_perimeter,
            links_col_perimeter,
            tile_row,
            tile_col,
        ) = future.result()
        with lock:
            global_state.update_perimeters(
                tile_row,
                tile_col,
                flow_acc_perimeter,
                flow_dir_perimeter,
                links_row_perimeter,
                links_col_perimeter,
            )
            flow_acc_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            flow_acc_tile.from_numpy(flow_acc)
            flow_acc_tile.write(output_band)
            task_queue.get()
            # Report progress as tiles complete
            chunk_counter[0] += 1
            tracker.callback(message=f"Chunk {chunk_counter[0]}/{total_chunks}")

    tracker.update(1, step_name="Calculate local")

    # Process each tile in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for flow_dir_tile in raster_chunker(input_band, chunk_size):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            future = executor.submit(
                calculate_flow_accumulation_tile,
                flow_dir_tile.data,
                flow_dir_tile.row,
                flow_dir_tile.col,
            )
            future.add_done_callback(handle_flow_acc_tile_result)

    # Wait for all tasks to complete
    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()

    # Calculate global accumulation
    tracker.update(2, step_name="Calculate global accumulation")
    source_index, source_value, exit_offsets, inject_index, inject_value = (
        _compute_upstream_injections(
            upstream_sources, input_band, chunk_size, global_state, np.int64
        )
    )
    global_acc, global_offset, _ = global_state.calculate_global_accumulation(
        inject_index, inject_value
    )
    _apply_upstream_sources(global_offset, exit_offsets)

    chunk_counter_finalize = [0]  # Use list for mutability in closure

    def handle_finalize_flow_acc_result(future):
        """Handle the result of finalizing flow accumulation for a tile."""
        with lock:
            try:
                flow_acc, tile_row, tile_col = future.result()
                flow_acc_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
                flow_acc_tile.from_numpy(flow_acc)
                flow_acc_tile.write(output_band)
            except Exception as e:
                print("Warning: Error finalizing flow accumulation for a tile", e)
            finally:
                task_queue.get()
                # Report progress as tiles complete
                chunk_counter_finalize[0] += 1
                tracker.callback(
                    message=f"Chunk {chunk_counter_finalize[0]}/{total_chunks}"
                )

    tracker.update(3, step_name="Finalize accumulation")

    # Finalize flow accumulation for each tile in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for flow_acc_tile in raster_chunker(output_band, chunk_size, lock=lock):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            flow_dir_tile = RasterChunk(
                flow_acc_tile.row, flow_acc_tile.col, chunk_size, 0
            )
            flow_dir_tile.read(input_band)

            future = executor.submit(
                finalize_flow_accumulation,
                flow_acc_tile.data,
                flow_dir_tile.data,
                global_acc,
                global_offset,
                flow_acc_tile.row,
                flow_acc_tile.col,
                global_state,
                source_index,
                source_value,
            )
            future.add_done_callback(handle_finalize_flow_acc_result)

    # Wait for all finalization tasks to complete
    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()
    output_ds = None


def setup_datasets(input_path, output_path):
    """
    Set up input and output datasets for flow accumulation calculation.

    Args:
        input_path (str): Path to input flow direction raster.
        output_path (str): Path for output flow accumulation raster.

    Returns:
        tuple: Flow direction dataset, output dataset, and no data value.
    """
    flow_dir_ds = open_dataset(input_path)
    input_band = flow_dir_ds.GetRasterBand(1)
    no_data_value = input_band.GetNoDataValue()
    if no_data_value is None:
        raise ValueError("Input raster must have a no data value")
    geotransform = flow_dir_ds.GetGeoTransform()
    projection = flow_dir_ds.GetProjection()
    output_ds = create_dataset(
        output_path,
        FLOW_ACCUMULATION_NODATA,
        gdal.GDT_Int64,
        input_band.XSize,
        input_band.YSize,
        geotransform,
        projection,
    )
    return flow_dir_ds, output_ds, no_data_value


def init_global_state(flow_dir_ds, output_ds, no_data_value, chunk_size):
    """
    Initialize the global state for tiled flow accumulation processing.

    Args:
        flow_dir_ds (gdal.Dataset): Flow direction dataset.
        output_ds (gdal.Dataset): Output dataset.
        no_data_value (float): No data value for the raster.
        chunk_size (int): Size of each tile (chunk) in pixels.

    Returns:
        tuple: Global state object, input band, and output band.
    """
    input_band = flow_dir_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)
    n_chunks_row = math.ceil(input_band.YSize / chunk_size)
    n_chunks_col = math.ceil(input_band.XSize / chunk_size)
    global_state = GlobalState(n_chunks_row, n_chunks_col, chunk_size, no_data_value)
    return global_state, input_band, output_band


def setup_datasets_weighted(fdr_path: str, weights_path: str, output_path: str):
    """
    Set up input and output datasets for weighted flow accumulation calculation.

    Args:
        fdr_path (str): Path to input flow direction raster.
        weights_path (str): Path to input weights raster, co-registered with the flow direction raster.
        output_path (str): Path for output flow accumulation raster.

    Returns:
        tuple: Flow direction dataset, weights dataset, output dataset, flow direction
            no data value, and weights no data value.
    """
    flow_dir_ds = open_dataset(fdr_path)
    input_band = flow_dir_ds.GetRasterBand(1)
    no_data_value = input_band.GetNoDataValue()
    if no_data_value is None:
        raise ValueError("Input raster must have a no data value")

    weights_ds = open_dataset(weights_path)
    weights_band = weights_ds.GetRasterBand(1)
    weights_nodata = weights_band.GetNoDataValue()
    if weights_nodata is None:
        raise ValueError("Weights raster must have a no data value")

    geotransform = flow_dir_ds.GetGeoTransform()
    projection = flow_dir_ds.GetProjection()
    output_ds = create_dataset(
        output_path,
        WEIGHTED_ACCUMULATION_NODATA,
        gdal.GDT_Float64,
        input_band.XSize,
        input_band.YSize,
        geotransform,
        projection,
    )
    return flow_dir_ds, weights_ds, output_ds, no_data_value, weights_nodata


def init_global_state_weighted(flow_dir_ds, output_ds, no_data_value, chunk_size):
    """
    Initialize the weighted global state for tiled flow accumulation processing.

    Args:
        flow_dir_ds (gdal.Dataset): Flow direction dataset.
        output_ds (gdal.Dataset): Output dataset.
        no_data_value (float): No data value for the flow direction raster.
        chunk_size (int): Size of each tile (chunk) in pixels.

    Returns:
        tuple: Weighted global state object, input band, and output band.
    """
    input_band = flow_dir_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)
    n_chunks_row = math.ceil(input_band.YSize / chunk_size)
    n_chunks_col = math.ceil(input_band.XSize / chunk_size)
    global_state = WeightedGlobalState(
        n_chunks_row, n_chunks_col, chunk_size, no_data_value
    )
    return global_state, input_band, output_band


def _flow_accumulation_weighted_tiled(
    fdr_path: str,
    weights_path: str,
    output_path: str,
    chunk_size: int,
    weights_nodata_mode: str = "zero",
    progress_callback: ProgressCallback | None = None,
    upstream_sources: dict[tuple[int, int], int] | None = None,
) -> None:
    """
    Compute weighted flow accumulation using a tiled approach for large rasters.

    Weighted analog of `_flow_accumulation_tiled`: sums an optional weights
    raster over each cell's upstream contributing area instead of counting
    cells. The flow direction raster drives tile iteration (it is mandatory);
    the weights raster is read at matching tile coordinates alongside it,
    mirroring the two-raster lockstep pattern used by `_extract_streams_tiled`.

    Args:
        fdr_path (str): Path to the input flow direction raster.
        weights_path (str): Path to the input weights raster, co-registered
            with the flow direction raster.
        output_path (str): Path for the output flow accumulation raster.
        chunk_size (int): Size of each tile (chunk) in pixels.
        weights_nodata_mode (str): "zero" or "propagate", controlling how
            nodata weight cells are handled.
        progress_callback (ProgressCallback | None): Optional callback for progress updates.
            If None, the operation runs silently.

    Returns:
        None
    """
    nodata_mode = _parse_weights_nodata_mode(weights_nodata_mode)

    # Setup progress tracking
    if progress_callback is None:
        progress_callback = silent_callback
    tracker = ProgressTracker(
        progress_callback, "Weighted Flow Accumulation", total_steps=3
    )

    # Setup datasets
    flow_dir_ds, weights_ds, output_ds, no_data_value, weights_nodata = (
        setup_datasets_weighted(fdr_path, weights_path, output_path)
    )
    global_state, input_band, output_band = init_global_state_weighted(
        flow_dir_ds, output_ds, no_data_value, chunk_size
    )
    weights_band = weights_ds.GetRasterBand(1)

    # Setup for parallel processing
    max_workers = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    task_queue: queue.Queue[int] = queue.Queue(max_workers)
    lock = Lock()

    chunk_counter = [0]  # Use list for mutability in closure
    total_chunks = math.ceil(input_band.YSize / chunk_size) * math.ceil(
        input_band.XSize / chunk_size
    )

    def handle_flow_acc_tile_result(future):
        """Handle the result of a single tile weighted flow accumulation calculation."""
        (
            flow_acc,
            flow_acc_perimeter,
            flow_dir_perimeter,
            links_row_perimeter,
            links_col_perimeter,
            tile_row,
            tile_col,
        ) = future.result()
        with lock:
            global_state.update_perimeters(
                tile_row,
                tile_col,
                flow_acc_perimeter,
                flow_dir_perimeter,
                links_row_perimeter,
                links_col_perimeter,
            )
            flow_acc_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            flow_acc_tile.from_numpy(flow_acc)
            flow_acc_tile.write(output_band)
            task_queue.get()
            # Report progress as tiles complete
            chunk_counter[0] += 1
            tracker.callback(message=f"Chunk {chunk_counter[0]}/{total_chunks}")

    tracker.update(1, step_name="Calculate local")

    # Process each tile in parallel; flow direction drives tile iteration,
    # weights is read at matching tile coordinates alongside it
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for flow_dir_tile in raster_chunker(input_band, chunk_size):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            weights_tile = RasterChunk(
                flow_dir_tile.row, flow_dir_tile.col, chunk_size, 0
            )
            weights_tile.read(weights_band)
            future = executor.submit(
                calculate_flow_accumulation_tile_weighted,
                flow_dir_tile.data,
                weights_tile.data,
                weights_nodata,
                nodata_mode,
                flow_dir_tile.row,
                flow_dir_tile.col,
            )
            future.add_done_callback(handle_flow_acc_tile_result)

    # Wait for all tasks to complete
    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()

    # Calculate global accumulation
    tracker.update(2, step_name="Calculate global accumulation")
    source_index, source_value, exit_offsets, inject_index, inject_value = (
        _compute_upstream_injections(
            upstream_sources, input_band, chunk_size, global_state, np.float64
        )
    )
    global_acc, global_offset, global_acc_is_external = (
        global_state.calculate_global_accumulation(inject_index, inject_value)
    )
    _apply_upstream_sources(global_offset, exit_offsets)

    chunk_counter_finalize = [0]  # Use list for mutability in closure

    def handle_finalize_flow_acc_result(future):
        """Handle the result of finalizing weighted flow accumulation for a tile."""
        with lock:
            try:
                flow_acc, tile_row, tile_col = future.result()
                flow_acc_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
                flow_acc_tile.from_numpy(flow_acc)
                flow_acc_tile.write(output_band)
            except Exception as e:
                print(
                    "Warning: Error finalizing weighted flow accumulation for a tile",
                    e,
                )
            finally:
                task_queue.get()
                # Report progress as tiles complete
                chunk_counter_finalize[0] += 1
                tracker.callback(
                    message=f"Chunk {chunk_counter_finalize[0]}/{total_chunks}"
                )

    tracker.update(3, step_name="Finalize accumulation")

    # Finalize flow accumulation for each tile in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for flow_acc_tile in raster_chunker(output_band, chunk_size, lock=lock):
            while task_queue.full():
                time.sleep(0.1)
            task_queue.put(0)
            flow_dir_tile = RasterChunk(
                flow_acc_tile.row, flow_acc_tile.col, chunk_size, 0
            )
            flow_dir_tile.read(input_band)

            future = executor.submit(
                finalize_flow_accumulation_weighted,
                flow_acc_tile.data,
                flow_dir_tile.data,
                global_acc,
                global_offset,
                global_acc_is_external,
                flow_acc_tile.row,
                flow_acc_tile.col,
                global_state,
                source_index,
                source_value,
            )
            future.add_done_callback(handle_finalize_flow_acc_result)

    # Wait for all finalization tasks to complete
    while not task_queue.empty():
        time.sleep(0.1)

    output_band.FlushCache()
    output_ds.FlushCache()
    output_ds = None

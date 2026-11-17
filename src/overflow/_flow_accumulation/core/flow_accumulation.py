import math
import warnings

import numpy as np
from numba import njit  # type: ignore[attr-defined]
from osgeo import gdal, ogr, osr

from overflow._util.constants import (
    FLOW_ACCUMULATION_NODATA,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_EXTERNAL,
    FLOW_TERMINATES,
    NEIGHBOR_OFFSETS,
    WEIGHTED_ACCUMULATION_NODATA,
    WEIGHTS_NODATA_MODE_PROPAGATE,
    WEIGHTS_NODATA_MODE_ZERO,
)
from overflow._util.queue import Int64PairQueue as Queue
from overflow._util.raster import create_dataset


@njit
def get_next_cell(
    flow_direction: np.ndarray, row: int, col: int
) -> tuple[int, int, int]:
    """Return the next (downstream) row, column, and value in the flow direction raster.

    Args:
        flow_direction (np.ndarray): Flow direction raster
        row (int): The row of the current cell
        col (int): The col of the current cell

    Returns:
        tuple[int, int, int]: the (row, col, val) of the next (downstream) cell
    """
    fdr_value = flow_direction[row, col]
    # Safety check: prevent out-of-bounds access on NEIGHBOR_OFFSETS (length 8)
    # FLOW_DIRECTION_UNDEFINED = 8, FLOW_DIRECTION_NODATA = 9
    if fdr_value >= 8:
        return row, col, fdr_value
    d_row, d_col = NEIGHBOR_OFFSETS[fdr_value]
    next_row = row + d_row
    next_col = col + d_col
    rows, cols = flow_direction.shape
    is_outside_tile = (
        next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols
    )
    if not is_outside_tile:
        return next_row, next_col, flow_direction[next_row, next_col]
    return next_row, next_col, FLOW_DIRECTION_NODATA


@njit
def perimeter_indices(shape):
    """Return the indices of the perimeter cells of a 2D array.

    Parameters
    ----------
    shape : tuple
        The shape (rows, cols) of the array.

    Returns
    -------
    list
        List of (row, col) tuples representing perimeter indices.
    """
    rows, cols = shape
    indices = []
    for i in range(rows):
        indices.append((i, 0))
        indices.append((i, cols - 1))
    for j in range(1, cols - 1):
        indices.append((0, j))
        indices.append((rows - 1, j))
    return indices


@njit
def follow_path(flow_direction, row, col, links, tile_row=0, tile_col=0):
    """Follow the flow path from a cell on the perimeter of the tile and
    populate the links array with the row and col of the perimeter cell that each cell drains to.
    If the cell drains directly to the edge of the tile, the FLOW_EXTERNAL value is used.
    If the cell terminates within the tile, the FLOW_TERMINATES value is used.
    If the cell drains through the tile, the row and col of the perimeter cell it drains to are used.
    This is Algorithm 2 from https://arxiv.org/pdf/1608.04431.pdf R. Barnes

    Args:
        flow_direction (np.ndarray): Flow direction raster
        row (int): the row of the cell on the perimeter
        col (int): the col of the cell on the perimeter
        links (np.ndarray): 3D array of shape (rows, cols, 2)
        tile_row (int): tile row index, used only in warning messages
        tile_col (int): tile col index, used only in warning messages
    """
    init_row = row
    init_col = col
    # sentinel must not collide with out-of-tile coordinates like (-1, -1)
    prev_row = -9999
    prev_col = -9999
    rows, cols = flow_direction.shape
    # Cycle detection: no valid path should exceed total cells in tile
    max_iterations = rows * cols
    iterations = 0
    while True:
        iterations += 1
        if iterations > max_iterations:
            # Cycle detected - mark as terminating to prevent infinite loop
            links[init_row, init_col, 0] = FLOW_TERMINATES[0]
            links[init_row, init_col, 1] = FLOW_TERMINATES[1]
            print(
                "Warning: Cycle detected in flow direction data. tile:",
                tile_row,
                tile_col,
                "cell:",
                row,
                col,
            )
            break
        next_row, next_col, next_val = get_next_cell(flow_direction, row, col)
        if next_row == prev_row and next_col == prev_col:
            # The current cell and the previous cell point at each other:
            # a two-cell cycle. Terminate immediately instead of spinning
            # until max_iterations.
            links[init_row, init_col, 0] = FLOW_TERMINATES[0]
            links[init_row, init_col, 1] = FLOW_TERMINATES[1]
            print(
                "Warning: Cycle detected in flow direction data. tile:",
                tile_row,
                tile_col,
                "cell:",
                row,
                col,
            )
            break
        is_outside_tile = (
            next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols
        )
        if is_outside_tile:
            if row == init_row and col == init_col:
                # cell drains immediately to the edge of the tile
                links[init_row, init_col, 0] = FLOW_EXTERNAL[0]
                links[init_row, init_col, 1] = FLOW_EXTERNAL[1]
            else:
                # cell drains through the tile
                links[init_row, init_col, 0] = row
                links[init_row, init_col, 1] = col
            break
        if next_val in (FLOW_DIRECTION_NODATA, FLOW_DIRECTION_UNDEFINED):
            # cell terminates within the tile
            links[init_row, init_col, 0] = FLOW_TERMINATES[0]
            links[init_row, init_col, 1] = FLOW_TERMINATES[1]
            break
        prev_row, prev_col = row, col
        row, col = next_row, next_col


@njit
def single_tile_flow_accumulation(
    flow_direction: np.ndarray,
    create_links: bool = True,
    tile_row: int = 0,
    tile_col: int = 0,
    upstream_sources: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate flow accumulation for a single tile.
       This is Algorithm 1 from https://arxiv.org/pdf/1608.04431.pdf R. Barnes

    Args:
        flow_direction (np.ndarray): Flow direction raster with codes from FlowDirection constants

    Returns:
        tuple[np.ndarray, np.ndarray]: (flow_accumulation, links)
         - flow_accumulation: The flow accumulation raster
         - links: The link raster identifying where each perimeter cell drains to
    """
    # the output flow accumulation raster
    flow_accumulation = np.zeros_like(flow_direction, dtype=np.int64)
    if upstream_sources is not None and upstream_sources.size > 0:
        for i in range(upstream_sources.shape[0]):
            row = upstream_sources[i, 0]
            col = upstream_sources[i, 1]
            value = upstream_sources[i, 2]
            if 0 <= row < flow_direction.shape[0] and 0 <= col < flow_direction.shape[1]:
                if flow_direction[row, col] == FLOW_DIRECTION_NODATA:
                    continue
                flow_accumulation[row, col] += value
    # a dependency raster (number of neighbors that flow into each cell)
    inflow_count = np.zeros_like(flow_direction, dtype=np.uint8)

    # Calculate inflow count
    for row, col in np.ndindex(flow_direction.shape):
        value = flow_direction[row, col]
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if value == FLOW_DIRECTION_NODATA:
            flow_accumulation[row, col] = FLOW_ACCUMULATION_NODATA
            continue
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        inflow_count[next_row, next_col] += 1

    queue = Queue([(0, 0)])
    queue.pop()

    source_count = 0
    if upstream_sources is not None and upstream_sources.size > 0:
        source_count = upstream_sources.shape[0]
    source_rows = np.empty((source_count, 2), dtype=np.int64)
    valid_source_count = 0
    if upstream_sources is not None and upstream_sources.size > 0:
        for i in range(upstream_sources.shape[0]):
            row = upstream_sources[i, 0]
            col = upstream_sources[i, 1]
            if 0 <= row < flow_direction.shape[0] and 0 <= col < flow_direction.shape[1]:
                source_rows[valid_source_count, 0] = row
                source_rows[valid_source_count, 1] = col
                valid_source_count += 1

    # Populate initial queue with any injected upstream sources and then with
    # cells that have no inflow.
    for i in range(valid_source_count):
        row = source_rows[i, 0]
        col = source_rows[i, 1]
        if flow_accumulation[row, col] != FLOW_ACCUMULATION_NODATA:
            queue.push((row, col))

    for row, col in np.ndindex(inflow_count.shape):
        _, _, next_value = get_next_cell(flow_direction, row, col)
        is_source = False
        for i in range(valid_source_count):
            if source_rows[i, 0] == row and source_rows[i, 1] == col:
                is_source = True
                break
        if (
            inflow_count[row, col] == 0
            and flow_accumulation[row, col] != FLOW_ACCUMULATION_NODATA
            and not is_source
        ):
            queue.push((row, col))

    # main loop
    while queue:
        row, col = queue.pop()
        flow_accumulation[row, col] += 1
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        flow_accumulation[next_row, next_col] += flow_accumulation[row, col]
        inflow_count[next_row, next_col] -= 1
        if inflow_count[next_row, next_col] == 0:
            queue.push((next_row, next_col))

    if not create_links:
        return flow_accumulation, None  # type: ignore[return-value]
    # links is a 3d numpy array containing the row and col on the
    # perimeter that each cell utlimatly drains to.
    # Only cells on the perimeter of the tile are considered.
    # Cells that drain directly to the edge of the tile are
    # marked with FLOW_EXTERNAL. Cells that terminate within the tile
    # are marked with FLOW_TERMINATES. Cells that drain through the tile
    # are marked with the row and col of the perimeter cell they drain to.
    links = np.empty(
        shape=(flow_direction.shape[0], flow_direction.shape[1], 2), dtype=np.int64
    )
    for row, col in perimeter_indices(flow_direction.shape):
        follow_path(flow_direction, row, col, links, tile_row, tile_col)
    return flow_accumulation, links


@njit
def single_tile_flow_accumulation_weighted(
    flow_direction: np.ndarray,
    weights: np.ndarray,
    weights_nodata: float,
    nodata_mode: int,
    create_links: bool = True,
    tile_row: int = 0,
    tile_col: int = 0,
    upstream_sources: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate weighted flow accumulation for a single tile: instead of
       counting cells, sums the weights raster over each cell's upstream
       contributing area. This is Algorithm 1 from
       https://arxiv.org/pdf/1608.04431.pdf R. Barnes, generalized to
       accumulate an arbitrary per-cell weight instead of 1.

    Args:
        flow_direction (np.ndarray): Flow direction raster with codes from FlowDirection constants
        weights (np.ndarray): Per-cell weight raster, same shape as flow_direction
        weights_nodata (float): The nodata value used in the weights raster
        nodata_mode (int): WEIGHTS_NODATA_MODE_ZERO or WEIGHTS_NODATA_MODE_PROPAGATE,
            controlling how nodata weight cells are handled

    Returns:
        tuple[np.ndarray, np.ndarray]: (flow_accumulation, links)
         - flow_accumulation: The weighted flow accumulation raster (float64)
         - links: The link raster identifying where each perimeter cell drains to
    """
    # substitute nodata weight cells with 0.0 (contribute nothing) or NaN
    # (poison this cell and everything downstream of it), once per tile,
    # so the main loop below stays a simple, branch-free `+=`
    effective_weights = np.empty(weights.shape, dtype=np.float64)
    for row, col in np.ndindex(weights.shape):
        value = weights[row, col]
        if value == weights_nodata:
            if nodata_mode == WEIGHTS_NODATA_MODE_PROPAGATE:
                effective_weights[row, col] = np.nan
            else:
                effective_weights[row, col] = 0.0
        else:
            effective_weights[row, col] = value

    # the output flow accumulation raster
    flow_accumulation = np.zeros_like(flow_direction, dtype=np.float64)
    if upstream_sources is not None and upstream_sources.size > 0:
        for i in range(upstream_sources.shape[0]):
            row = upstream_sources[i, 0]
            col = upstream_sources[i, 1]
            value = upstream_sources[i, 2]
            if 0 <= row < flow_direction.shape[0] and 0 <= col < flow_direction.shape[1]:
                if flow_direction[row, col] == FLOW_DIRECTION_NODATA:
                    continue
                flow_accumulation[row, col] += value
    # a dependency raster (number of neighbors that flow into each cell)
    inflow_count = np.zeros_like(flow_direction, dtype=np.uint8)

    # Calculate inflow count
    for row, col in np.ndindex(flow_direction.shape):
        value = flow_direction[row, col]
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if value == FLOW_DIRECTION_NODATA:
            flow_accumulation[row, col] = WEIGHTED_ACCUMULATION_NODATA
            continue
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        inflow_count[next_row, next_col] += 1

    queue = Queue([(0, 0)])
    queue.pop()

    source_count = 0
    if upstream_sources is not None and upstream_sources.size > 0:
        source_count = upstream_sources.shape[0]
    source_rows = np.empty((source_count, 2), dtype=np.int64)
    valid_source_count = 0
    if upstream_sources is not None and upstream_sources.size > 0:
        for i in range(upstream_sources.shape[0]):
            row = upstream_sources[i, 0]
            col = upstream_sources[i, 1]
            if 0 <= row < flow_direction.shape[0] and 0 <= col < flow_direction.shape[1]:
                source_rows[valid_source_count, 0] = row
                source_rows[valid_source_count, 1] = col
                valid_source_count += 1

    # Populate initial queue with any injected upstream sources and then with
    # cells that have no inflow.
    for i in range(valid_source_count):
        row = source_rows[i, 0]
        col = source_rows[i, 1]
        if not np.isnan(flow_accumulation[row, col]):
            queue.push((row, col))

    for row, col in np.ndindex(inflow_count.shape):
        _, _, next_value = get_next_cell(flow_direction, row, col)
        is_source = False
        for i in range(valid_source_count):
            if source_rows[i, 0] == row and source_rows[i, 1] == col:
                is_source = True
                break
        if (
            inflow_count[row, col] == 0
            and not np.isnan(flow_accumulation[row, col])
            and not is_source
        ):
            queue.push((row, col))

    # main loop
    while queue:
        row, col = queue.pop()
        flow_accumulation[row, col] += effective_weights[row, col]
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        flow_accumulation[next_row, next_col] += flow_accumulation[row, col]
        inflow_count[next_row, next_col] -= 1
        if inflow_count[next_row, next_col] == 0:
            queue.push((next_row, next_col))

    if not create_links:
        return flow_accumulation, None  # type: ignore[return-value]
    # links is a 3d numpy array containing the row and col on the
    # perimeter that each cell utlimatly drains to.
    # Only cells on the perimeter of the tile are considered.
    # Cells that drain directly to the edge of the tile are
    # marked with FLOW_EXTERNAL. Cells that terminate within the tile
    # are marked with FLOW_TERMINATES. Cells that drain through the tile
    # are marked with the row and col of the perimeter cell they drain to.
    links = np.empty(
        shape=(flow_direction.shape[0], flow_direction.shape[1], 2), dtype=np.int64
    )
    for row, col in perimeter_indices(flow_direction.shape):
        follow_path(flow_direction, row, col, links, tile_row, tile_col)
    return flow_accumulation, links


def _parse_upstream_sources_field(
    layer: ogr.Layer, upstream_sources_field: str | None = None
) -> str:
    """Resolve the field name used to store integer accumulation values."""
    field_count = layer.GetLayerDefn().GetFieldCount()
    if field_count == 0:
        raise ValueError(
            "Upstream source dataset does not contain any attribute fields; "
            "provide a field with integer accumulation values."
        )

    candidates = [
        "value",
        "accumulation",
        "upstream_value",
        "source_value",
        "fac",
        "weight",
        "contribution",
    ]
    if upstream_sources_field is not None:
        candidates = [upstream_sources_field, *candidates]

    for candidate in candidates:
        if candidate is None:
            continue
        for field_index in range(field_count):
            field_defn = layer.GetLayerDefn().GetFieldDefn(field_index)
            if field_defn.GetName().lower() == candidate.lower():
                return field_defn.GetName()

    if upstream_sources_field is not None:
        raise ValueError(
            f"Upstream source field '{upstream_sources_field}' was not found in the "
            "input vector dataset."
        )

    raise ValueError(
        "Upstream source dataset does not contain a recognized integer value "
        "field. Supply upstream_sources_field with a field name like 'value'."
    )


def _snap_point_to_boundaries(
    x: float,
    y: float,
    transform: tuple[float, ...],
    rows: int,
    cols: int,
    flow_direction: np.ndarray | None = None,
) -> tuple[int, int]:
    """Map a point to the raster cell that contains it.

    This is a simple point-in-raster lookup for upstream inflow sources. If the
    point lies outside the raster extent, it is treated as invalid and is
    ignored upstream.
    """
    inv_transform = gdal.InvGeoTransform(transform)
    if inv_transform == (None, None, None, None, None, None):
        raise ValueError("Could not invert flow direction raster geotransform.")

    cell_x, cell_y = gdal.ApplyGeoTransform(inv_transform, x, y)
    row = int(math.floor(cell_y))
    col = int(math.floor(cell_x))

    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise ValueError(
            f"Upstream source at point ({x}, {y}) is outside the flow direction raster; "
            "ignoring this source."
        )

    return row, col


def _coerce_upstream_value(feature: ogr.Feature, field_name: str) -> int:
    """Validate that an upstream source value is integer-like."""
    field_index = feature.GetFieldIndex(field_name)
    if field_index < 0:
        raise ValueError(f"Upstream source field '{field_name}' not found.")

    value = feature.GetField(field_name)
    if value is None:
        raise ValueError(f"Upstream source feature has a null value in field '{field_name}'.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Upstream source field '{field_name}' must contain numeric values."
        ) from exc

    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(
            f"Upstream source field '{field_name}' must contain integer values only; "
            f"floating-point sources are not allowed."
        )
    return int(numeric_value)


def _read_upstream_sources(
    fdr_path: str,
    upstream_sources_path: str,
    upstream_sources_field: str | None = None,
) -> dict[tuple[int, int], int]:
    """Read and snap an OGR point source dataset to nearest edge cells."""
    fdr_ds = gdal.Open(fdr_path)
    if fdr_ds is None:
        raise ValueError(f"Could not open flow direction raster '{fdr_path}'.")

    source_ds = ogr.Open(upstream_sources_path)
    if source_ds is None:
        raise ValueError(f"Could not open upstream source dataset '{upstream_sources_path}'.")

    source_layer = source_ds.GetLayer()
    if source_layer is None:
        raise ValueError(f"No vector layer found in '{upstream_sources_path}'.")

    geom_type = source_layer.GetGeomType()
    if geom_type not in {
        ogr.wkbPoint,
        ogr.wkbPoint25D,
        ogr.wkbMultiPoint,
        ogr.wkbMultiPoint25D,
    }:
        raise ValueError(
            "Upstream source dataset must be a point vector layer; "
            f"found geometry type {geom_type}."
        )

    field_name = _parse_upstream_sources_field(source_layer, upstream_sources_field)
    transform = fdr_ds.GetGeoTransform()
    inv_transform = gdal.InvGeoTransform(transform)
    if inv_transform == (None, None, None, None, None, None):
        raise ValueError("Could not invert flow direction raster geotransform.")
    inv_transform = tuple(inv_transform)
    rows = fdr_ds.RasterYSize
    cols = fdr_ds.RasterXSize
    fdr_band = fdr_ds.GetRasterBand(1)
    fdr_array = fdr_band.ReadAsArray()

    source_map: dict[tuple[int, int], int] = {}
    layer_srs = source_layer.GetSpatialRef()
    raster_projection = fdr_ds.GetProjection()
    raster_srs = None
    if raster_projection and raster_projection.strip():
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster_projection)

    for feature in source_layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        geom_type = geom.GetGeometryType()
        if geom_type in (ogr.wkbPoint, ogr.wkbPoint25D):
            points = [(geom.GetX(), geom.GetY())]
        elif geom_type in (ogr.wkbMultiPoint, ogr.wkbMultiPoint25D):
            points = []
            for i in range(geom.GetGeometryCount()):
                part = geom.GetGeometryRef(i)
                if part is not None:
                    points.append((part.GetX(), part.GetY()))
        else:
            raise ValueError(
                "Each upstream source feature must be a point or multipoint geometry. "
                f"Feature {feature.GetFID()} is geometry type {geom_type}."
            )

        for x, y in points:
            if layer_srs is not None and raster_srs is not None:
                if not layer_srs.IsSame(raster_srs):
                    source_srs = osr.SpatialReference()
                    source_srs.ImportFromWkt(layer_srs.ExportToWkt())
                    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    raster_srs.SetAxisMappingStrategy(
                        osr.OAMS_TRADITIONAL_GIS_ORDER
                    )
                    transform_to_raster = osr.CoordinateTransformation(
                        source_srs, raster_srs
                    )
                    x, y, _ = transform_to_raster.TransformPoint(x, y)

            try:
                snapped_row, snapped_col = _snap_point_to_boundaries(
                    x, y, transform, rows, cols, fdr_array
                )
            except ValueError:
                warnings.warn(
                    "Upstream source at point "
                    f"({x}, {y}) is outside the flow direction raster; "
                    "ignoring this source.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if not (0 <= snapped_row < rows and 0 <= snapped_col < cols):
                continue

            source_value = _coerce_upstream_value(feature, field_name)
            cell_key = (snapped_row, snapped_col)
            previous = source_map.get(cell_key, 0)
            if cell_key in source_map:
                warnings.warn(
                    "Duplicate upstream source contribution snapped to the same boundary cell "
                    f"({snapped_row}, {snapped_col}); summing values.",
                    UserWarning,
                    stacklevel=2,
                )
            source_map[cell_key] = previous + source_value

    source_ds = None
    fdr_ds = None
    return source_map


def _parse_weights_nodata_mode(mode: str) -> int:
    """Translate the public string weights_nodata_mode API into the internal
    int flags consumed by the njit-compiled weighted accumulation functions.

    Args:
        mode (str): "zero" or "propagate"

    Returns:
        int: WEIGHTS_NODATA_MODE_ZERO or WEIGHTS_NODATA_MODE_PROPAGATE

    Raises:
        ValueError: If mode is not "zero" or "propagate"
    """
    if mode == "zero":
        return WEIGHTS_NODATA_MODE_ZERO
    if mode == "propagate":
        return WEIGHTS_NODATA_MODE_PROPAGATE
    raise ValueError(
        f"Invalid weights_nodata_mode: {mode!r}. Must be 'zero' or 'propagate'."
    )


def _upstream_sources_to_array(
    upstream_sources: dict[tuple[int, int], int] | None,
) -> np.ndarray:
    """Convert the public source dictionary into a 2D int64 array for Numba."""
    if not upstream_sources:
        return np.empty((0, 3), dtype=np.int64)
    rows = np.empty((len(upstream_sources), 3), dtype=np.int64)
    for index, ((row, col), value) in enumerate(upstream_sources.items()):
        rows[index, 0] = row
        rows[index, 1] = col
        rows[index, 2] = value
    return rows


def _flow_accumulation_weighted(
    fdr_path: str,
    weights_path: str,
    output_fac_path: str,
    weights_nodata_mode: str = "zero",
    upstream_sources: dict[tuple[int, int], int] | None = None,
) -> None:
    """
    Generates a weighted flow accumulation raster from a flow direction raster
    and a co-registered weights raster.

    Parameters
    ----------
    fdr_path : str
        Path to the input flow direction raster file.
    weights_path : str
        Path to the input weights raster file, co-registered with the flow direction raster.
    output_fac_path : str
        Path to the output flow accumulation raster file.
    weights_nodata_mode : str
        "zero" (nodata weight cells contribute 0) or "propagate" (nodata weight
        cells poison their own and all downstream accumulation with NaN).

    Returns
    -------
    None
    """
    nodata_mode = _parse_weights_nodata_mode(weights_nodata_mode)
    fdr_ds = gdal.Open(fdr_path)
    projection = fdr_ds.GetProjection()
    transform = fdr_ds.GetGeoTransform()

    fdr_ds_band = fdr_ds.GetRasterBand(1)
    fdr_array = fdr_ds_band.ReadAsArray().astype("uint8")

    weights_ds = gdal.Open(weights_path)
    weights_band = weights_ds.GetRasterBand(1)
    weights_nodata = weights_band.GetNoDataValue()
    if weights_nodata is None:
        raise ValueError("Weights raster must have a no data value")
    weights_array = weights_band.ReadAsArray().astype(np.float64)

    fac_ds = create_dataset(
        output_fac_path,
        WEIGHTED_ACCUMULATION_NODATA,
        gdal.GDT_Float64,
        fdr_ds.RasterXSize,
        fdr_ds.RasterYSize,
        transform,
        projection,
    )
    fac_band = fac_ds.GetRasterBand(1)
    source_array = _upstream_sources_to_array(upstream_sources)
    fac_array, _ = single_tile_flow_accumulation_weighted(
        fdr_array,
        weights_array,
        weights_nodata,
        nodata_mode,
        False,
        upstream_sources=source_array,
    )
    fac_band.WriteArray(fac_array)
    fac_band.FlushCache()
    fac_ds.FlushCache()
    fac_band = None
    fac_ds = None
    fdr_ds_band = None
    fdr_ds = None
    weights_band = None
    weights_ds = None


def _flow_accumulation(
    fdr_path: str,
    output_fac_path: str,
    upstream_sources: dict[tuple[int, int], int] | None = None,
) -> None:
    """
    Generates a flow accumulation raster from a flow direction raster.

    Parameters
    ----------
    fdr_path : str
        Path to the input flow direction raster file.
    output_fac_path : str
        Path to the output flow accumulation raster file.
    upstream_sources : dict[tuple[int, int], int] | None
        Optional source-cell contributions to add before the standard
        propagation pass.

    Returns
    -------
    None
    """
    fdr_ds = gdal.Open(fdr_path)
    projection = fdr_ds.GetProjection()
    transform = fdr_ds.GetGeoTransform()

    fdr_ds_band = fdr_ds.GetRasterBand(1)
    fdr_array = fdr_ds_band.ReadAsArray().astype("uint8")
    fac_ds = create_dataset(
        output_fac_path,
        FLOW_ACCUMULATION_NODATA,
        gdal.GDT_Int64,
        fdr_ds.RasterXSize,
        fdr_ds.RasterYSize,
        transform,
        projection,
    )
    fac_band = fac_ds.GetRasterBand(1)
    source_array = _upstream_sources_to_array(upstream_sources)
    fac_array, _ = single_tile_flow_accumulation(
        fdr_array,
        False,
        upstream_sources=source_array,
    )
    fac_band.WriteArray(fac_array)
    fac_band.FlushCache()
    fac_ds.FlushCache()
    fac_band = None
    fac_ds = None
    fdr_ds_band = None
    fdr_ds = None

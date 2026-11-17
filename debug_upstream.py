import numpy as np
from osgeo import gdal, ogr, osr
from overflow import accumulation
from overflow._flow_accumulation.core.flow_accumulation import _read_upstream_sources, single_tile_flow_accumulation

fdr = np.zeros((1,5), dtype=np.ubyte)
fdr_path = '/vsimem/test_debug_fdr.tif'
ds = gdal.GetDriverByName('GTiff').Create(fdr_path, 5, 1, 1, gdal.GDT_Byte)
band = ds.GetRasterBand(1)
band.WriteArray(fdr)
band.SetNoDataValue(9)
ds.SetGeoTransform((0,1,0,1,0,-1))
ds = None

src_path = '/vsimem/test_debug_src.geojson'

drv = ogr.GetDriverByName('GeoJSON')
d = drv.CreateDataSource(src_path)
srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
l = d.CreateLayer('layer', srs=srs, geom_type=ogr.wkbPoint)
fd = ogr.FieldDefn('value', ogr.OFTInteger)
l.CreateField(fd)
feat = ogr.Feature(l.GetLayerDefn())
g = ogr.Geometry(ogr.wkbPoint)
g.AddPoint(1.3, 0.5)
feat.SetGeometry(g)
feat.SetField('value', 5)
l.CreateFeature(feat)
d = None

print('read sources', _read_upstream_sources(fdr_path, src_path))
print('kernel direct', single_tile_flow_accumulation(fdr, False, upstream_sources=np.array([[0,0,5]], dtype=np.int64))[0])
out = '/vsimem/test_debug_out.tif'
accumulation(fdr_path, out, chunk_size=0, upstream_sources_path=src_path)
arr = gdal.Open(out).GetRasterBand(1).ReadAsArray()
print('accum output', arr)

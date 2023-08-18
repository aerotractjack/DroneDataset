import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, box, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
import torch
from PIL import Image

def test_paths():
    tif = "/home/aerotract/software/DroneDataset/data/potlach101/101ShanghaiKnight_Orthomosaic_export_TueJul18150704535595.tif"
    points = "/home/aerotract/software/DroneDataset/data/potlach101/Buffered_Trees_1.geojson"
    aoi = "/home/aerotract/software/DroneDataset/data/potlach101/AOI_1.geojson"
    return tif, aoi, points

class AeroDataset:

    def __init__(self, tif_path, aoi_path=None, label_path=None, **kw):
        self.tif_path = tif_path
        self.src_img_path = tif_path
        self.aoi_path = aoi_path
        self.label_path = label_path
        self.aoi_path = aoi_path
        if self.aoi_path is not None:
            self.cropped_path = self.crop_image_to_file()
            self.src_img_path = self.cropped_path
        self.kw = kw

    @property
    def crs(self):
        with rasterio.open(self.src_img_path) as src:
            return src.crs

    def crop_image_to_file(self):
        tif_path = Path(self.src_img_path)
        aoi_path = Path(self.aoi_path)
        crop_path = tif_path.with_stem(tif_path.stem + "_" + aoi_path.stem)
        aoi_gdf = gpd.read_file(self.aoi_path)
        aoi_gdf = aoi_gdf.to_crs(self.crs)
        aoi_geometry = [mapping(aoi_gdf.geometry.values[0])]
        with rasterio.open(tif_path) as src:
            out_image, out_transform = mask(src, aoi_geometry, crop=True)
            out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
        with rasterio.open(crop_path, "w", **out_meta) as dest:
            dest.write(out_image)
        return crop_path
    
    def window_pixel_to_latlong(self, window):
        with rasterio.open(self.src_img_path) as src:
            crs = src.crs
            l, b, r, t = src.window_bounds(window)
        window_geom = box(l, b, r, t)
        window_geom = gpd.GeoSeries(window_geom, crs=crs).to_crs(crs).values[0]
        return window_geom

    def window_x(self, window):
        with rasterio.open(self.src_img_path) as src:
            tile = src.read(window=window)
        width = tile.shape[1]
        height = tile.shape[2]
        pad_width = window.width - width
        pad_height = window.height - height
        if pad_width > 0 or pad_height > 0:
            tile = np.pad(tile, ((0, 0), (0, pad_width), (0, pad_height)), mode='constant')
        tile = tile.transpose(1, 2, 0)
        return tile
    
    def window_y(self, window, window_latlong, latlong=False):
        labeled_gdf = gpd.read_file(self.label_path)
        tile_box = window_latlong
        crs = self.crs
        with rasterio.open(self.src_img_path) as src:
            l, b, r, t = src.window_bounds(window)
            intersecting_polygons = labeled_gdf[labeled_gdf.geometry.intersects(tile_box)]
            intersecting_polygons = intersecting_polygons.to_crs(crs)
            window_transform = src.window_transform(window)
        tile_polygon = Polygon([(l, b), (r, b), (r, t), (l, t), (l, b)])

        def contain_ratio(geom):
            if not geom.is_valid:
                geom = geom.buffer(0)
            intersection = geom.intersection(tile_polygon)
            return intersection.area / geom.area
        
        def to_pixel_coordinates(geom):
            if geom.geom_type == 'Polygon':
                return Polygon([~window_transform * coord for coord in geom.exterior.coords])
            elif geom.geom_type == 'MultiPolygon':
                return MultiPolygon([Polygon([~window_transform * coord for coord in polygon.exterior.coords]) for polygon in geom.geoms])
        
        intersecting_polygons['containment'] = intersecting_polygons['geometry'].apply(contain_ratio)
        if intersecting_polygons.shape[0] != 0:
            if not latlong:
                intersecting_polygons['geometry'] = intersecting_polygons['geometry'].apply(to_pixel_coordinates)
            fltr = float(self.kw.get("containment_ratio", 0.1))
            intersecting_polygons = intersecting_polygons[intersecting_polygons['containment'] > fltr]
        return intersecting_polygons.to_crs(self.crs)

    def window_data(self, window, latlong=False):
        x = self.window_x(window)
        if self.label_path is None:
            return x
        else:
            window_latlong = self.window_pixel_to_latlong(window)
            y = self.window_y(window, window_latlong, latlong=latlong)
            return x, y

    def iter(self, sampler, latlong=False):
        for window, _ in sampler:
            yield self.window_data(window, latlong=latlong)

    def plot(self, sampler_cls, **kw):
        for window_data in self.iter(sampler_cls, **kw):
            if self.label_path is None:
                x = window_data
                y = None
            else:
                x, y = window_data
            plt.figure()
            plt.imshow(x)
            if y is not None:
                for geometry in y['geometry']:
                    if geometry.geom_type == 'Polygon':
                        x, y = geometry.exterior.xy
                        plt.plot(x, y, color='blue')
                    elif geometry.geom_type == 'MultiPolygon':
                        for polygon in geometry.geoms:
                            x, y = polygon.exterior.xy
                            plt.plot(x, y, color='blue')
            plt.show()

class TorchVisionAeroDataset(Dataset):

    def __init__(self, aerods, sampler):
        self.aerods = aerods
        self.sampler = sampler

    def reset(self):
        self.sampler.reset()

    def __len__(self):
        return len(self.sampler)
    
    def process_x(self, x):
        return Image.fromarray(x[:,:,:3])
    
    def process_y(self, y, image_id=0):
        target = {}
        g = y["geometry"]
        target["boxes"] = torch.FloatTensor(g.apply(lambda x: x.bounds).tolist())
        labels = (y["class_id"] + 1 - y["class_id"].min()).tolist()
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([image_id], dtype=torch.uint8)
        target["area"] = torch.FloatTensor(g.apply(lambda x: x.area).tolist())
        return target

    def __getitem__(self, idx):
        window, window_transform = next(self.sampler)
        data = self.aerods.window_data(window)
        if self.aerods.label_path is None:
            return self.process_x(data)
        return window_transform, (self.process_x(data[0]), self.process_y(data[1], image_id=idx))
        
    def pixel_to_latlong(self, window_transform, boxes):
        polygons = []
        for box in boxes:
            x0, y0, x1, y1 = box
            polygon_coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            polygons.append(Polygon(polygon_coords))
        latlong_polygons = [Polygon([(window_transform * coord) for coord in polygon.exterior.coords]) for polygon in polygons]
        return gpd.GeoDataFrame(geometry=latlong_polygons, crs=self.aerods.crs)

if __name__ == "__main__":
    from samplers import RandomSampler

    tif, aoi, points = test_paths()
    
    ds = AeroDataset(tif, aoi, points)
    sampler = RandomSampler(ds.src_img_path, **{"tile_size": 128, "n": 11})
    td = TorchVisionAeroDataset(ds, sampler)

    df = []
    for window_transform, (x, y) in td:
        ll = td.pixel_to_latlong(window_transform, y["boxes"])
        df.extend(ll.values[:,0].tolist())
    gdf = gpd.GeoDataFrame(geometry=df, crs=ds.crs)
    gdf.to_file("test.geojson")
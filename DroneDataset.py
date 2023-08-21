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
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
import samplers

def test_paths():
    tif = "/home/aerotract/software/DroneDataset/data/potlach101/101ShanghaiKnight_Orthomosaic_export_TueJul18150704535595.tif"
    points = "/home/aerotract/software/DroneDataset/data/potlach101/Buffered_Trees_1.geojson"
    aoi = "/home/aerotract/software/DroneDataset/data/potlach101/AOI_1.geojson"
    return tif, aoi, points

class DroneDataset:
    ''' dataset for geotiff images + geojson labels '''

    def __init__(self, tif_path, aoi_path=None, label_path=None, **kw):
        # path to input image, assume this is cropped to an aoi
        self.tif_path = tif_path
        # assuming the tif is already cropped, it becomes our source image
        self.src_img_path = tif_path
        # record the aoi and label paths (if given)
        self.aoi_path = aoi_path
        self.label_path = label_path
        # we're given an aoi path, so this image is not cropped
        if self.aoi_path is not None:
            # crop the image and record the cropped path
            self.cropped_path = self.crop_image_to_file()
            # this cropped path becomes the source image
            self.src_img_path = self.cropped_path
        # record keyword args
        self.kw = kw

    @property
    def crs(self):
        # quick property to access our source image crs
        with rasterio.open(self.src_img_path) as src:
            return src.crs
        
    @property
    def num_classes(self):
        if self.label_path is None:
            return None
        gdf = gpd.read_file(self.label_path)
        return len(gdf["class_id"].unique())

    def crop_image_to_file(self):
        # use the given aoi to crop our tif and save it to a file in the
        # same directory as the tif with a new name
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
        # convert a windows pixel coords to lat/long
        with rasterio.open(self.src_img_path) as src:
            crs = src.crs
            l, b, r, t = src.window_bounds(window)
        window_geom = box(l, b, r, t)
        window_geom = gpd.GeoSeries(window_geom, crs=crs).to_crs(crs).values[0]
        return window_geom

    def window_x(self, window):
        # access the image content within a window, padding if necessary
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
        # access the label polygons within a window
        # calculate the percent that each polygon overlaps the aoi (1 if completely inside, 0 if outside)
        #   and filter out results with ratio <= kw.get("containment_ratio")
        # convert latlong polygons to pixel coords (unless latlong=True)
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
        # return the image and labels (if present) for each a given window
        x = self.window_x(window)
        if self.label_path is None:
            return x
        else:
            window_latlong = self.window_pixel_to_latlong(window)
            y = self.window_y(window, window_latlong, latlong=latlong)
            return x, y

    def iter(self, sampler):
        # generator to yield all window data
        for window, _ in sampler:
            yield self.window_data(window)

    def plot(self, sampler_cls, **kw):
        # helper method to visualize generated windows/labels from a given WindowSampler
        for window_data in self.iter(sampler_cls, **kw):
            if self.label_path is None:
                x = window_data
                y = None
            else:
                x, y = window_data
            plt.figure()
            plt.imshow(x)
            if y is None:
                plt.show()
                return
            for geometry in y['geometry']:
                if geometry.geom_type == 'Polygon':
                    x, y = geometry.exterior.xy
                    plt.plot(x, y, color='blue')
                    continue
                for polygon in geometry.geoms:
                    x, y = polygon.exterior.xy
                    plt.plot(x, y, color='blue')
            plt.show()

class TorchVisionDroneDataset(Dataset):
    ''' torchvision-compatible class for DroneDataset+WindowSampler '''

    def __init__(self, aerods, sampler, x_transform=None, y_transform=None):
        # dataset to iterate over
        self.aerods = aerods
        # WindowSampler object to sample windows
        self.sampler = sampler
        # fn to transform images
        self.x_transform = self._x_transform if x_transform is None else x_transform
        # fn to transform targets
        self.y_transform = self._y_transform if y_transform is None else y_transform

    @property
    def num_classes(self):
        return self.aerods.num_classes

    def __len__(self):
        # required by Dataset class
        return len(self.sampler)
    
    def process_x(self, x):
        # process input image from DroneDataset
        return Image.fromarray(x[:,:,:3])
    
    def process_y(self, y, image_id=0):
        # process input boxes from DroneDataset
        target = {}
        g = y["geometry"]
        target["boxes"] = torch.FloatTensor(g.apply(lambda _y: _y.bounds).tolist())
        labels = (y["class_id"] + 1 - y["class_id"].min()).tolist()
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([image_id], dtype=torch.uint8)
        target["area"] = torch.FloatTensor(g.apply(lambda x: x.area).tolist())
        return target
    
    def _x_transform(self, x):
        # base transform for images, convert from PIL to float32 tensor
        x = pil_to_tensor(x)
        x = convert_image_dtype(x, torch.float32)
        return x
    
    def _y_transform(self, y):
        # base transform for targets
        if np.isnan((y['boxes']).numpy()).any() or y['boxes'].shape == torch.Size([0]):
            y['boxes'] = torch.zeros((0,4),dtype=torch.float32)
        return y
    
    def process_data(self, window, window_transform, idx=0):
        data = self.aerods.window_data(window)
        if self.aerods.label_path is None:
            return self.x_transform(self.process_x(data))
        x = self.process_x(data[0])
        y = self.process_y(data[1], image_id=idx)
        x = self.x_transform(x)
        y = self.y_transform(y)
        return x, y
    
    def __getitem__(self, idx):
        wts, xs, ys = [], [], []
        windows, window_transforms = next(self.sampler)
        for i in range(len(windows)):
            wts.append(window_transforms[i])
            if self.aerods.label_path is not None:
                x, y = self.process_data(windows[i], window_transforms[i], idx=idx)
                xs.append(x)
                ys.append(y)
            else:
                x = self.process_data(windows[i], window_transforms[i], idx=idx)
                xs.append(x)
        if len(ys) > 0:
            return wts, xs, ys
        return wts, xs
        
    def _pixel_to_latlong(self, window_transform, boxes):
        polygons = []
        for box in boxes:
            x0, y0, x1, y1 = box
            polygon_coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            polygons.append(Polygon(polygon_coords))
        latlong_polygons = [Polygon([(window_transform * coord) for coord in polygon.exterior.coords]) for polygon in polygons]
        return gpd.GeoDataFrame(geometry=latlong_polygons, crs=self.aerods.crs)
    
    def pixel_to_latlong(self, window_transform, boxes):
        if isinstance(window_transform, list):
            out = []
            for i in range(len(window_transform)):
                out.append(self._pixel_to_latlong(window_transform[i], boxes[i]))
            return out
        return self._pixel_to_latlong(window_transform, boxes)
    
    @classmethod
    def TrainingDataset(cls, tif=None, aoi=None, points=None, sampler="RandomSampler", **kw):
        ds = DroneDataset(tif, aoi, points)
        sampler = getattr(samplers, sampler)(ds.src_img_path, **kw)
        return cls(ds, sampler)
    
    @classmethod
    def EvalDataset(cls, tif=None, aoi=None, points=None, sampler="StrideSampler", **kw):
        ds = DroneDataset(tif, aoi, points)
        sampler = getattr(samplers, sampler)(ds.src_img_path, **kw)
        return cls(ds, sampler)
    
    @classmethod
    def PredictionDataset(cls, tif=None, aoi=None, **kw):
        ds = DroneDataset(tif, aoi)
        sampler = samplers.StrideSampler(ds.src_img_path, **kw)
        samplers.StrideSampler.windows_to_file(ds.src_img_path, **kw)
        return cls(ds, sampler)

def get_sample_dataset(**kw):
    from samplers import RandomSampler
    tif, aoi, points = test_paths()
    ds = DroneDataset(tif, aoi, points)
    sampler = RandomSampler(ds.src_img_path, **kw)
    td = TorchVisionDroneDataset(ds, sampler)
    return td

if __name__ == "__main__":
    from samplers import RandomSampler
    ds = DroneDataset(*test_paths())
    RandomSampler.windows_to_file(ds.src_img_path, tile_size=128, n=1, batch_size=1)
    ds.num_classes
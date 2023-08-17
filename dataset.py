import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, box, Polygon, MultiPolygon
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from pathlib import Path
import matplotlib.pyplot as plt

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
        with rasterio.open(tif_path) as src:
            aoi_gdf = aoi_gdf.to_crs(src.crs)
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

    def calc_rows_cols(self, tile_size=128, overlap=.05):
        step_size = tile_size * (1 - overlap)
        with rasterio.open(self.src_img_path) as src:
            rows = src.height // step_size + 1
            cols = src.width // step_size + 1
        return int(rows), int(cols)

    def window(self, row, col, tile_size=128, overlap=.05):
        overlap = tile_size * overlap 
        step_size = tile_size - overlap
        row = row * step_size
        col = col * step_size
        with rasterio.open(self.src_img_path) as src:
            print(tile_size, src.width, col)
            window_width = int(min(tile_size, src.width - col))
            window_height = int(min(tile_size, src.height - row))
            print(window_width, window_height)
            window = Window(col_off=col, row_off=row, \
                            width=window_width, height=window_height)
            return window
        
    def window_latlong(self, window):
        crs = self.crs
        with rasterio.open(self.src_img_path) as src:
            l, b, r, t = src.window_bounds(window)
        window_geom = box(l, b, r, t)
        window_geom = gpd.GeoSeries(window_geom, crs=crs).to_crs(crs).values[0]
        return window_geom

    def window_data(self, row, col, tile_size=128, latlong=False):
        window = self.window(row, col, tile_size=tile_size)
        with rasterio.open(self.src_img_path) as src:
            tile = src.read(window=window)
        pad_width = tile_size - window.width
        pad_height = tile_size - window.height
        if pad_width > 0 or pad_height > 0:
            tile = np.pad(tile, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant')
        tile = tile.transpose(1, 2, 0)
        if tile.size == 0 or tile.sum() == 0:
            return None
        if self.label_path is None:
            return tile
        return tile, self.window_labels(window, latlong)
            
    def window_labels(self, window, latlong=False):
        labeled_gdf = gpd.read_file(self.label_path)
        tile_box = self.window_latlong(window)
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
            fltr = float(self.kw.get("containment_ratio", 0.1))
            intersecting_polygons = intersecting_polygons[intersecting_polygons['containment'] > fltr]
        if not latlong:
            intersecting_polygons['geometry'] = intersecting_polygons['geometry'].apply(to_pixel_coordinates)
        return intersecting_polygons.to_crs(self.crs)

    def iter(self, tile_size=128, windows_only=False, overlap=.05):
        num_rows, num_cols = self.calc_rows_cols(tile_size=tile_size, overlap=overlap)
        for row in range(num_rows):
            for col in range(num_cols):
                print("=======")
                print(row, col)
                if windows_only:
                    w = self.window(row, col, tile_size=tile_size, overlap=overlap)
                    yield self.window_latlong(w)
                    continue
                tile = self.window_data(row, col, tile_size)
                if tile is None:
                    continue
                yield tile
                
    def plot_x(self, tile_size=128, overlap=.05):
        for tile in self.iter(tile_size, overlap=overlap):
            if self.label_path is not None:
                tile, _ = tile
            plt.figure()
            plt.imshow(tile)
            plt.show()

    def plot_xy(self, tile_size=128, overlap=.05):
        for image, polygons in self.iter(tile_size, overlap=overlap):
            plt.figure()
            plt.imshow(image)
            for geometry in polygons['geometry']:
                if geometry.geom_type == 'Polygon':
                    x, y = geometry.exterior.xy
                    plt.plot(x, y, color='blue')
                elif geometry.geom_type == 'MultiPolygon':
                    for polygon in geometry.geoms:
                        x, y = polygon.exterior.xy
                        plt.plot(x, y, color='blue')
            plt.axis('off') 
            plt.show()
        return plt
    
    def plot(self, tile_size=128, overlap=.05):
        for data in self.iter(tile_size=tile_size, overlap=overlap):
            image, polygons = None, None
            if self.label_path is None:
                image = data
            else:
                image, polygons = data
            plt.figure()
            plt.imshow(image)
            if polygons is not None:
                for geometry in polygons['geometry']:
                    if geometry.geom_type == 'Polygon':
                        x, y = geometry.exterior.xy
                        plt.plot(x, y, color='blue')
                    elif geometry.geom_type == 'MultiPolygon':
                        for polygon in geometry.geoms:
                            x, y = polygon.exterior.xy
                            plt.plot(x, y, color='blue')
            plt.axis('off')
            plt.show()

    def save_windows(self, tile_size=128, overlap=.05, row=None, col=None):
        crs = self.crs
        gdf = []
        if row is None or col is None:
            for w in self.iter(tile_size=128, windows_only=True, overlap=overlap):
                gdf.append(w)
        else:
            w = self.window(row, col, tile_size=tile_size, overlap=overlap) 
            gdf.append(w)
        gdf = GeoDataFrame(geometry=gdf, crs=crs)
        p = self.src_img_path
        filename = p.with_stem(p.stem + "_" + "windows").with_suffix(".geojson")
        gdf.to_file(filename, driver="GeoJSON")
        return filename, gdf

def test_paths():
    tif = "/home/aerotract/software/data_playground/data/potlach101/101ShanghaiKnight_Orthomosaic_export_TueJul18150704535595.tif"
    points = "/home/aerotract/software/data_playground/data/potlach101/Buffered_Trees_1.geojson"
    aoi = "/home/aerotract/software/data_playground/data/potlach101/AOI_1.geojson"
    return tif, aoi, points

if __name__ == "__main__":

    tif, aoi, points = test_paths()

    cds = AeroDataset(tif, aoi, points)
    for tile, label in cds.iter():
        pass
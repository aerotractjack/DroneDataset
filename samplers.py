from geopandas import GeoDataFrame
import numpy as np
import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

##############
# base class #
##############

class WindowSampler:
    ''' base class for creating sampler objects. these are responsible for iterating
    through our dataset and generating windows to tile our images/labels '''

    def __init__(self, src_img_path, **kw):
        # path to the image to tile
        self.src_img_path = Path(src_img_path)
        with rasterio.open(self.src_img_path) as src:
            # record width and height
            self.width = src.width
            self.height = src.height
        # pixel size of tiles
        self.tile_size = kw.get("tile_size", 128)
        # batch size
        self.batch_size = kw.get("batch_size", 4)
        # how many batches of tiles to yield
        self.n = kw.get("n", None)
        # yield tiles with pixel or latlong coordinates
        self.latlong = kw.get("latlong", False)
        # current iteration
        self.current = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        return self
    
    def __next__(self):
        # only yield a certain amount
        if self.n is not None and self.current >= self.n:
            raise StopIteration
        self.current += 1
        return self.getnext()
    
    def getnext(self):
        return (0,0)
    
    def window_latlong(self, window):
        # convert a windows pixel coords to lat/long
        with rasterio.open(self.src_img_path) as src:
            crs = src.crs
            l, b, r, t = src.window_bounds(window)
        window_geom = box(l, b, r, t)
        window_geom = gpd.GeoSeries(window_geom, crs=crs).to_crs(crs).values[0]
        return window_geom

    def build_window(self, row, col):
        # build a window centered around the given a pixel row and col
        # return the window (used to tile the image and labels) and the 
        # window_transform, used to convert pixels to latlong
        window_width = self.tile_size
        window_height = self.tile_size
        window = Window(col_off=col, row_off=row, \
                        width=window_width, height=window_height)
        with rasterio.open(self.src_img_path) as src:
            window_transform = src.window_transform(window)
        if self.latlong:
            return self.window_latlong(window), window_transform
        return window, window_transform
    
    @classmethod
    def windows_to_file(cls, src_img_path, **kw):
        # create an instance of this sampler and write the generated windows 
        # to a geojson for validation in qgis
        self = cls(src_img_path, latlong=True, **kw)
        windows = []
        for w, _ in self:
            windows.extend(w)
        with rasterio.open(self.src_img_path) as src:
            crs = src.crs
        gdf = GeoDataFrame(geometry=windows, crs=crs)
        p = self.src_img_path
        filename = p.with_stem(p.stem + "_" + "windows").with_suffix(".geojson")
        gdf.to_file(filename, driver="GeoJSON")
        return self
    
##################
# random sampler #
##################

class RandomSampler(WindowSampler):
    ''' sample windows randomly within the image '''

    def __init__(self, src_img_path, **kw):
        # if `n` is none, this will be infinite and the user must break their own loop
        kw["n"] = kw.get("n", 50)
        super().__init__(src_img_path, **kw)

    def getnext(self):
        # sample points within an image
        windows, wts = [], []
        for b in range(self.batch_size):
            row = np.random.randint(0, self.height - (self.tile_size // 2))
            col = np.random.randint(0, self.width - (self.tile_size // 2))
            w, wt = self.build_window(row, col)
            windows.append(w)
            wts.append(wt)
        return windows, wts

##################
# stride sampler #
##################

class StrideSampler(WindowSampler):

    def __init__(self, src_img_path, **kw):
        super().__init__(src_img_path, **kw)
        self.current_row = -self.tile_size // 4
        self.current_col = -self.tile_size // 4
        self.overlap = kw.get("overlap", 0)
        self.adj_tile_size = self.tile_size * (1 - self.overlap)
        self.n = self.calc_n()
        print(self.n)

    def calc_n(self):
        # needs work
        return int((self.height * self.width) / (self.adj_tile_size**2)) + 1

    def getnext(self, rc_only=False):
        cr, cc = self.current_row, self.current_col
        if self.current_col < (self.width - self.adj_tile_size):
            self.current_col += self.adj_tile_size
        else:
            self.current_col = -self.adj_tile_size // 4
            self.current_row += self.adj_tile_size
        if rc_only:
            return cr, cc
        w, wt = self.build_window(cr, cc)
        return [w], [wt]
            
if __name__ == "__main__":
    from DroneDataset import DroneDataset, test_paths
    ds = DroneDataset(*test_paths())
    StrideSampler.windows_to_file(ds.src_img_path)
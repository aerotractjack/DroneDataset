from geopandas import GeoDataFrame
import numpy as np
import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path

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
        # how many tiles to yield
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
    
    def build_window(self, row, col):
        # build a window centered around the given a pixel row and col
        # return the window (used to tile the image and labels) and the 
        # window_transform, used to convert pixels to latlong
        window_width = self.tile_size
        window_height = self.tile_size
        window = Window(col_off=col, row_off=row, \
                        width=window_width, height=window_height)
        if self.latlong:
            return self.window_latlong(window)
        with rasterio.open(self.src_img_path) as src:
            window_transform = src.window_transform(window)
        return window, window_transform
    
    @classmethod
    def windows_to_file(cls, src_img_path, n, **kw):
        # create an instance of this sampler and write the generated windows 
        # to a geojson for validation in qgis
        self = cls(src_img_path, n=n, latlong=True, **kw)
        windows = []
        for w in self:
            windows.append(w)
        with rasterio.open(self.src_img_path) as src:
            crs = src.crs
        gdf = GeoDataFrame(geometry=windows, crs=crs)
        p = self.src_img_path
        filename = p.with_stem(p.stem + "_" + "windows").with_suffix(".geojson")
        gdf.to_file(filename, driver="GeoJSON")
        return self
    
class RandomSampler(WindowSampler):
    ''' sample windows randomly within the image '''

    def __init__(self, src_img_path, **kw):
        # if `n` is none, this will be infinite and the user must break their own loop
        super().__init__(src_img_path, **kw)

    def getnext(self):
        # sample points within an image
        row = np.random.randint(0, self.height - (self.tile_size // 2))
        col = np.random.randint(0, self.width - (self.tile_size // 2))
        return self.build_window(row, col)

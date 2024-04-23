import glob
import rasterio
import numpy as np


if __name__ == "__main__":
    all_path = glob.glob('/media/yusin/Elements/DFC2022/*altas.tif')
    all_color = []
    for alt_path in all_path:
        with rasterio.open(alt_path) as data:
            real = data.read([1, 2, 3])
            colours = np.unique(np.rollaxis(real, 0, 3).reshape(-1, 3), axis=0)
            all_color.extend(colours)
            print(colours)
    print(list(set(all_color)))

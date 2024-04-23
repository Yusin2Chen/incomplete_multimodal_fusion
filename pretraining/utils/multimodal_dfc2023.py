import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import cv2
import torch

def resiz_4pl(img, size):
    imgs = np.zeros((img.shape[0], size[0], size[1]))
    for i in range(img.shape[0]):
        per_img = np.squeeze(img[i, :, :])
        per_img = cv2.resize(per_img, size, interpolation=cv2.INTER_AREA)
        imgs[i, :, :] = per_img
    return imgs

def normalization(data):
    if torch.is_tensor(data):
        _range = torch.max(data) - torch.min(data)
        return (data - torch.min(data)) / _range
    else:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

# standar
rgb_MEAN = np.array([81.29692, 87.93711, 72.041306])
rgb_STD  = np.array([39.61512, 35.407978, 35.84708])

def normalize_rgb(imgs):
    for i in range(3):
        imgs[i, :, :] = (imgs[i, :, :] - rgb_MEAN[i]) / rgb_STD[i]
    return imgs


sar_MEAN = np.array([-7.9447875, ])
sar_STD  = np.array([2.777256, ])

def normalize_sar(imgs):
    for i in range(1):
        imgs[i, :, :] = (imgs[i, :, :] - sar_MEAN[i]) / sar_STD[i]
    return imgs


dem_MEAN = np.array([5.0160093, ])
dem_STD  = np.array([7.6128364, ])

def normalize_dem(imgs):
    for i in range(1):
        imgs[i, :, :] = (imgs[i, :, :] - dem_MEAN[i]) / dem_STD[i]
    return imgs

# data augmenttaion
class RandomCrop(object):
    """给定图片，随机裁剪其任意一个和给定大小一样大的区域.

    Args:
        output_size (tuple or int): 期望裁剪的图片大小。如果是 int，将得到一个正方形大小的图片.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample, unlabeld=True):

        if unlabeld:
            s1, s2, dem, id = sample['s1'], sample['s2'], sample['dem'], sample['id']
            lc = None
        else:
            s1, s2, dem, lc, id = sample['s1'], sample['s2'], sample['dem'], sample['label'], sample['id']

        _, h, w = s2.shape
        new_h, new_w = self.output_size
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        #print(s1.shape, s2.shape, dem.shape)
        s1 = s1[:, top: top + new_h, left: left + new_w]
        s2 = s2[:, top: top + new_h, left: left + new_w]
        dem = dem[:, top: top + new_h, left: left + new_w]


        # load label
        if unlabeld:
            return {'s1': s1, 's2': s2, 'dem': dem, 'id': id}
        else:
            lc = lc[top: top + new_h, left: left + new_w]
            return {'s1': s1, 's2': s2, 'dem': dem, 'label': lc, 'id': id}



# util function for reading dsm data
def load_dsm(path):
    # load labels
    with rasterio.open(path) as data:
        dsm = data.read(1)
    #print('dsm:', dsm.shape, dsm.min(), dsm.max())
    dsm = np.nan_to_num(dsm)
    #dsm = np.clip(dsm, 0, 1000)
    dsm = dsm[np.newaxis, :, :]
    dsm = resiz_4pl(dsm, (256, 256))
    dsm = dsm.astype(np.float32)
    dsm = (dsm - dsm.mean()) / np.sqrt(dsm.var() + 1e-6)
    #dsm = normalization(dsm)

    return dsm


# util function for reading dsm data
def load_rgb(path):
    # load labels
    with rasterio.open(path) as data:
        rgb = data.read()
    #print('rgb:', rgb.shape, rgb.min(), rgb.max())
    rgb = np.nan_to_num(rgb)
    rgb = resiz_4pl(rgb, (256, 256))
    rgb = rgb.astype(np.float32)
    rgb = normalize_rgb(rgb)

    return rgb


# util function for reading s1 data
def load_sar(path):
    with rasterio.open(path) as data:
        sar = data.read()
    #print('sar:', sar.shape, sar.min(), sar.max())
    sar = 10 * np.log10(sar + 0.0000001)
    sar = np.clip(sar, -25, 0)
    sar = np.nan_to_num(sar)
    sar = resiz_4pl(sar, (256, 256))
    sar = sar.astype(np.float32)
    sar = normalize_sar(sar)

    return sar


# util function for reading lc data
def load_lc(path):
    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    return lc


def load_rgb_sar_dsm(sample, use_rgb, use_sar, use_dsm, unlabeled=False):
    # load rgb data
    if use_rgb:
        rgb = load_rgb(sample["rgb"])
    else:
        rgb = None

    # load sar data
    if use_sar:
        sar = load_sar(sample["sar"])
    else:
        sar = None

    # load dsm data
    if use_dsm:
        dsm = load_dsm(sample["dsm"])
    else:
        dsm = None

    # load label
    if unlabeled:
        return {'s1': sar, 's2': rgb, 'dem': dsm, 'id': sample["id"]}
    else:
        lc = load_lc(sample["lc"])
        return {'s1': sar, 's2': rgb, 'dem': dsm, 'label': lc, 'id': sample["id"]}


class DFC2023(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 use_rgb=True,
                 use_sar=True,
                 use_dsm=True,
                 unlabeled=True,
                 transform=False,
                 crop_size=32):
        """Initialize the dataset"""

        # inizialize
        super(DFC2023, self).__init__()

        self.use_rgb = use_rgb
        self.use_sar = use_sar
        self.use_dsm = use_dsm
        self.unlabeled = unlabeled

        # define transform
        if transform:
            self.transform = RandomCrop(crop_size)
        else:
            self.transform = None
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        self.samples = []
        rgb_locations = glob.glob(os.path.join(path, f"{'rgb'}/*.tiff"), recursive=True)
        for rgb_loc in tqdm(rgb_locations, desc="[Load]"):
            sar_loc = rgb_loc.replace("rgb", "sar")
            dsm_loc = rgb_loc.replace("rgb", "dsm")
            if self.unlabeled:
                self.samples.append(
                    {"rgb": rgb_loc, "sar": sar_loc, "dsm": dsm_loc, "id": os.path.basename(rgb_loc)})
            else:
                lc_loc = rgb_loc.replace("rgb", "lc")
                self.samples.append(
                    {"lc": lc_loc, "rgb": rgb_loc, "sar": sar_loc, "dsm": dsm_loc, "id": os.path.basename(rgb_loc)})

        print("loaded", len(self.samples), "samples from the dfc2020 subset")

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        data_sample = load_rgb_sar_dsm(sample, self.use_rgb, self.use_sar, self.use_dsm, unlabeled=self.unlabeled)
        if self.transform:
            return self.transform(data_sample)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)



if __name__ == "__main__":
    print("\n\nDFC2023 test")
    data_dir = '../../DFC2023/track2/images'
    ds = DFC2023(data_dir, use_rgb=True, use_sar=True, use_dsm=True, unlabeled=True, transform=False)
    num = len(ds)
    s1_meanlist = []
    s1_stdlist = []
    s21_meanlist = []
    s21_stdlist = []
    s22_meanlist = []
    s22_stdlist = []
    s23_meanlist = []
    s23_stdlist = []
    dem_meanlist = []
    dem_stdlist = []
    for i in range(num):
        s = ds.__getitem__(i)
        s1_meanlist.append(s["s1"].mean())
        s1_stdlist.append(s["s1"].std())
        s21_meanlist.append(s["s2"][0, :, :].mean())
        s21_stdlist.append(s["s2"][0, :, :].std())
        s22_meanlist.append(s["s2"][1, :, :].mean())
        s22_stdlist.append(s["s2"][1, :, :].std())
        s23_meanlist.append(s["s2"][2, :, :].mean())
        s23_stdlist.append(s["s2"][2, :, :].std())
        dem_meanlist.append(s["dem"].mean())
        dem_stdlist.append(s["dem"].std())
        #print(s1_meanlist)
    s1_meanarr = np.array(s1_meanlist)
    s1_stdarr = np.array(s1_stdlist)
    s21_meanarr = np.array(s21_meanlist)
    s21_stdarr = np.array(s21_stdlist)
    s22_meanarr = np.array(s22_meanlist)
    s22_stdarr = np.array(s22_stdlist)
    s23_meanarr = np.array(s23_meanlist)
    s23_stdarr = np.array(s23_stdlist)
    dem_meanarr = np.array(dem_meanlist)
    dem_stdarr = np.array(dem_stdlist)
    print("loaded", len(ds), "samples from the dfc2023 subset")
    print('s1:', s1_meanarr.mean(), s1_stdarr.mean(), '21:', s21_meanarr.mean(), s21_stdarr.mean(), '22:',
          s22_meanarr.mean(), s22_stdarr.mean(), '23:', s23_meanarr.mean(), s23_stdarr.mean(),
          'dem:', dem_meanarr.mean(), dem_stdarr.mean())

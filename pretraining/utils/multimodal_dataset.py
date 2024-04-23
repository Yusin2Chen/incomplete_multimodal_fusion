import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm
import torch.utils.data as data

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def normalize_S2(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i, :,:] - S2_MEAN[i]) / S2_STD[i]
    return imgs

S1_MEAN = np.array([-9.020017, -15.73008])
S1_STD  = np.array([3.5793820, 3.671725])

def normalize_S1(imgs):
    for i in range(2):
        imgs[i,:,:] = (imgs[i, :,:] - S1_MEAN[i]) / S1_STD[i]
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

    def __call__(self, sample, unlabeld=True, superpixel=True):

        if unlabeld:
            image, id = sample['image'], sample['id']
            lc = None
        else:
            image, lc, id = sample['image'], sample['label'], sample['id']

        _, h, w = image.shape
        new_h, new_w = self.output_size
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top: top + new_h, left: left + new_w]

        if superpixel:
            segments = sample["segments"]
            segments = segments[top: top + new_h, left: left + new_w]
        else:
            segments = None
        # index
        index = sample["index"]
        index = index[:, top: top + new_h, left: left + new_w]

        # load label
        if unlabeld:
            return {'image': image, 'segments': segments, 'index': index, 'id': id}
        else:
            lc = lc[top: top + new_h, left: left + new_w]
            return {'image': image, 'segments': segments, 'index': index, 'label': lc, 'id': id}


# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]


# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 = normalize_S2(s2)
    return s2


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 = normalize_S1(s1)
    return s1

# util function for reading dsm data
def load_dsm(path):
    # load labels
    with rasterio.open(path) as data:
        dsm = data.read(1)
    dsm = dsm.astype(np.float32)
    dsm = np.nan_to_num(dsm)
    dsm = np.clip(dsm, -25, 0)
    dsm = normalize_DSM(dsm)

    return dsm

# util function for reading dsm data
def load_rgb(path):
    # load labels
    with rasterio.open(path) as data:
        rgb = data.read()
    rgb = rgb.astype(np.float32)
    rgb = np.nan_to_num(rgb)
    rgb = normalize_RGB(rgb)

    return rgb

# util function for reading s1 data
def load_sar(path):
    with rasterio.open(path) as data:
        sar = data.read()
    sar = sar.astype(np.float32)
    sar = np.nan_to_num(sar)
    sar = np.clip(sar, -25, 0)
    sar = normalize_SAR(sar)
    return sar


# util function for reading lc data
def load_lc(path):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    return lc

# this function for classification and most important is for weak supervised
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr, use_dsm, unlabeled=False):

    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
    else:
        img = None

    # load s1 data
    if use_s1:
        if use_s2:
            img = np.concatenate((img, load_s1(sample["s1"])), axis=0)
        else:
            img = load_s1(sample["s1"])


    # load label
    if unlabeled:
        return {'image': img, 'id': sample["id"]}
    else:
        lc = load_lc(sample["lc"])
        return {'image': img, 'label': lc, 'id': sample["id"]}


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
                 train_index=None,
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
        rgb_locations = glob.glob(os.path.join(path, f"{'rgb'}/*.tif"), recursive=True)
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
        data_sample = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                                  self.use_s2lr, self.use_dsm, unlabeled=self.unlabeled)
        if self.transform:
            return self.transform(data_sample)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


class DFC2020(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 use_dsm=False,
                 unlabeled=True,
                 transform=False,
                 train_index=None,
                 crop_size=32):
        """Initialize the dataset"""

        # inizialize
        super(DFC2020, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
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
        train_list = []
        train_list += [x for x in os.listdir(path)]
        train_list = [x for x in train_list if "s1_" in x]

        self.samples = []
        for folder in train_list:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            for s2_loc in tqdm(s2_locations, desc="[Load]"):
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                se_loc = s2_loc.replace("tif", "npy").replace("s2_", "se_").replace("_s2_", "_se_")
                lc_loc = s2_loc.replace("_s2_", "_dfc_").replace("s2_", "dfc_")
                self.samples.append(
                    {"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "se": se_loc, "id": os.path.basename(lc_loc)})

        print("loaded", len(self.samples), "samples from the dfc2020 subset")

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        data_sample = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                                  self.use_s2lr, self.use_dsm, unlabeled=self.unlabeled)
        if self.transform:
            return self.transform(data_sample)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

if __name__ == "__main__":
    print("\n\nDFC2023 test")
    data_dir = '../DFC2023'
    ds = DFC2023(data_dir)
    s, index = ds.__getitem__(0)
    print("id:", s["id"], "\n", "input shape:", s["s1"].shape)

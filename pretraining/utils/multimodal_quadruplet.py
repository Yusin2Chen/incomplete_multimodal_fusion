import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import torch
import cv2

def resiz_4pl(img, size):
    imgs = np.zeros((img.shape[0], size[0], size[1]))
    for i in range(img.shape[0]):
        per_img = np.squeeze(img[i, :, :])
        per_img = cv2.resize(per_img, size, interpolation=cv2.INTER_AREA)
        imgs[i, :, :] = per_img
    return imgs


num_classes = 28
colormaps = [[0, 0, 0], [128, 0, 0], [191, 0, 0], [255, 64, 64], [255, 128, 128], [255, 191, 191],
            [204, 102, 102], [204, 77, 242], [149, 149, 149], [179, 179, 179],
            [89, 89, 89], [230, 204, 204], [230, 204, 230], [115, 77, 55], [185, 165, 110],
            [135, 69, 69], [140, 220, 0], [175, 210, 165], [255, 255, 168],
            [242, 166, 77], [230, 230, 77], [255, 230, 77], [242, 204, 128], [0, 140, 0],
            [204, 242, 77], [204, 255, 204], [166, 166, 255], [128, 242, 230]]
classess = ['Continuous Urban fabric (S.L. > 80%)', 'Discontinuous Dense Urban Fabric (S.L.: 50% - 80%)', 'Discontinuous Medium Density Urban Fabric (S.L.: 30% - 50%)',
            'Discontinuous Low Density Urban Fabric (S.L.: 10% - 30%)', 'Discontinuous very low density urban fabric (S.L. < 10%)', 'Isolated Structures',
            'Industrial, commercial, public, military and private units', 'Fast transit roads and associated land', 'Other roads and associated land',
            'Railways and associated land', 'Port areas', 'Airports', 'Mineral extraction and dump sites', 'Construction sites', 'Land without current use',
            'Green urban areas', 'Sports and leisure facilities', 'Arable land (annual crops)', 'Permanent crops', 'Pastures' ,'Complex and mixed cultivation patterns',
            'Orchards', 'Forests', 'Herbaceous vegetation associations', 'Open spaces with little or no vegetations', 'Wetlands', 'Water']

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(colormaps):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Index2Color(pred):
    colormap = np.asarray(colormaps, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[0, :, :] * 256 + data[1, :, :]) * 256 + data[2, :, :]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)

def normalization(data):
    if torch.is_tensor(data):
        _range = torch.max(data) - torch.min(data) + 0.000001
        return (data - torch.min(data)) / _range
    else:
        _range = np.max(data) - np.min(data) + 0.000001
        return (data - np.min(data)) / _range

S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def normalize_S2(imgs):
    for i in range(4):
        imgs[i, :, :] = (imgs[i, :, :] - S2_MEAN[i]) / S2_STD[i]
    return imgs

S1_MEAN = np.array([-9.020017, -15.73008])
S1_STD  = np.array([3.5793820, 3.671725])

def normalize_S1(imgs):
    for i in range(2):
        imgs[i, :, :] = (imgs[i, :, :] - S1_MEAN[i]) / S1_STD[i]
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
            s1, s2, dem, dnw, id = sample['s1'], sample['s2'], sample['dem'], sample['dnw'], sample['id']
            lc = None
        else:
            s1, s2, dem, dnw, lc, id = sample['s1'], sample['s2'], sample['dem'], sample['dnw'], sample['label'], sample['id']

        _, h, w = s2.shape
        new_h, new_w = self.output_size
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        #print(s1.shape, s2.shape, dem.shape)
        s1 = s1[:, top: top + new_h, left: left + new_w]
        s2 = s2[:, top: top + new_h, left: left + new_w]
        dem = dem[:, top: top + new_h, left: left + new_w]
        dnw = dnw[top: top + new_h, left: left + new_w]


        # load label
        if unlabeld:
            return {'s1': s1, 's2': s2, 'dem': dem, 'dnw': dnw, 'id': id}
        else:
            lc = lc[top: top + new_h, left: left + new_w]
            return {'s1': s1, 's2': s2, 'dem': dem, 'dnw': dnw, 'label': lc, 'id': id}



# util function for reading dsm data
def load_dem(path):
    # load labels
    with rasterio.open(path) as data:
        dsm = data.read([1])  # 有一个维度， 如果只写1则只有两维
    #print('dsm:', dsm.shape, dsm.min(), dsm.max())
    dsm = np.nan_to_num(dsm)
    dsm = np.clip(dsm, -100, 5000)
    #dsm = resiz_4pl(dsm, (256, 256))
    dsm = dsm.astype(np.float32)
    dsm = normalization(dsm)
    return dsm


# util function for reading s2 data
S2_BANDS_HR = [2, 3, 4, 8]
def load_s2(path):
    # load labels
    with rasterio.open(path) as data:
        s2 = data.read(S2_BANDS_HR)
    s2 = np.nan_to_num(s2)
    s2 = np.clip(s2, 0, 10000)
    #s2 = resiz_4pl(s2, (256, 256))
    s2 = s2.astype(np.float32)
    s2 = normalize_S2(s2)

    return s2


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read([1, 2])
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 25)
    #s1 = resiz_4pl(s1, (256, 256))
    s1 = s1.astype(np.float32)
    s1 = normalize_S1(s1)
    return s1


# util function for reading dnw data
def load_dnw(path):
    # load labels
    with rasterio.open(path) as data:
        dnw = data.read([10])  #这样就只有两维
        #dnw = resiz_4pl(dnw, (256, 256))
        dnw = dnw.squeeze().astype(np.int_)
    return dnw

# util function for reading lc data
def load_lc(path):
    # load labels
    with rasterio.open(path) as data:
        lc = data.read([1, 2, 3])
        lc = Color2Index(lc)
    return lc

def load_quadruplet(sample, use_s1, use_s2, use_dem, use_dnw, unlabeled=False):
    # load s1 data
    if use_s1:
        s1 = load_s1(sample["s1"])
    else:
        s1 = None

    # load s2 data
    if use_s2:
        s2 = load_s2(sample["s2"])
    else:
        s2 = None

    # load dem data
    if use_dem:
        dem = load_dem(sample["dem"])
    else:
        dem = None

    # load dnw data
    if use_dnw:
        dnw = load_dnw(sample["dnw"])
    else:
        dnw = None

    # load label
    if unlabeled:
        return {'s1': s1, 's2': s2, 'dem': dem, 'dnw': dnw, 'id': sample["id"]}
    else:
        lc = load_lc(sample["lc"])
        return {'s1': s1, 's2': s2, 'dem': dem, 'dnw': dnw, 'label': lc, 'id': sample["id"]}


class MyDataset(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 use_s1=True,
                 use_s2=True,
                 use_dem=True,
                 use_dnw=True,
                 unlabeled=True,
                 transform=False,
                 crop_size=32):
        """Initialize the dataset"""

        # inizialize
        super(MyDataset, self).__init__()

        self.use_s1 = use_s1
        self.use_s2 = use_s2
        self.use_dem = use_dem
        self.use_dnw = use_dnw
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
        train_list = []
        #for place in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13',
        #              'a14', 'a15', 'a16', 'a17', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
        #              'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20']:
        for place in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']:
            train_list += [os.path.join(place, x) for x in os.listdir(os.path.join(path, place))]
            train_list = [x for x in train_list if "s2_" in x]

        for folder in train_list:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            for s2_loc in tqdm(s2_locations, desc="[Load]"):
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                dem_loc = s2_loc.replace("_s2_", "_dem_").replace("s2_", "dem_")
                dnw_loc = s2_loc.replace("_s2_", "_dnw_").replace("s2_", "dnw_")
                if self.unlabeled and os.path.exists(s1_loc) and os.path.exists(dem_loc) and os.path.exists(dnw_loc):
                    self.samples.append(
                        {"s1": s1_loc, "s2": s2_loc, "dem": dem_loc, "dnw": dnw_loc, "id": os.path.basename(s2_loc)})
                elif os.path.exists(s1_loc) and os.path.exists(dem_loc) and os.path.exists(dnw_loc):
                    lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")
                    self.samples.append(
                        {"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "dem": dem_loc, "dnw": dnw_loc, "id": os.path.basename(s2_loc)})

            print("loaded", len(self.samples), "samples from the dfc2020 subset")



    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        data_sample = load_quadruplet(sample, self.use_s1, self.use_s2, self.use_dem, self.use_dnw, unlabeled=self.unlabeled)
        if self.transform:
            return self.transform(data_sample)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)



if __name__ == "__main__":
    print("\n\nDFC2023 test")
    data_dir = '../../DFC2023'
    ds = MyDataset(data_dir, use_s1=True, use_s2=True, use_dem=True, use_dnw=True, unlabeled=True, transform=False)
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

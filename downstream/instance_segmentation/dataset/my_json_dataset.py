import os
import json
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from .coco_utils import coco_remove_images_without_annotations, convert_coco_poly_mask
import cv2
import rasterio

def normalization(data):
    if torch.is_tensor(data):
        _range = torch.max(data) - torch.min(data)
        return (data - torch.min(data)) / _range
    else:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

# standar
rgb_MEAN = np.array([81.17236, 87.812256, 71.91678])
rgb_STD  = np.array([41.254555, 37.071312, 37.43461])

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

# util function for reading dsm data
def load_dsm(path):
    # load labels
    with rasterio.open(path) as data:
        dsm = data.read(1)
    #print('dsm:', dsm.shape, dsm.min(), dsm.max())
    dsm = dsm.astype(np.float32)
    dsm = np.nan_to_num(dsm)
    #dsm = np.clip(dsm, 0, 1000)
    dsm = dsm[np.newaxis, :, :]
    #dsm = normalize_dem(dsm)
    dsm = (dsm - dsm.mean()) / np.sqrt(dsm.var() + 1e-6)
    #dsm = normalization(dsm)
    return dsm

# util function for reading dsm data
def load_rgb(path):
    # load labels
    with rasterio.open(path) as data:
        rgb = data.read()
    #print('rgb:', rgb.shape, rgb.min(), rgb.max())
    rgb = rgb.astype(np.float32)
    rgb = np.nan_to_num(rgb)
    rgb = normalize_rgb(rgb)
    return rgb

# util function for reading s1 data
def load_sar(path):
    with rasterio.open(path) as data:
        sar = data.read()
    sar = 10 * np.log10(sar + 0.0000001)
    sar = np.clip(sar, -25, 0)
    #print('sar:', sar.shape, sar.min(), sar.max())
    sar = sar.astype(np.float32)
    sar = np.nan_to_num(sar)
    sar = normalize_sar(sar)
    return sar

def resiz_4pl(img, size):
    imgs = np.zeros((img.shape[0], size[0], size[1]))
    for i in range(img.shape[0]):
        per_img = np.squeeze(img[i, :, :])
        per_img = cv2.resize(per_img, size, interpolation=cv2.INTER_AREA)
        imgs[i, :, :] = per_img
    return imgs

class CocoDetection(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        dataset (string): train or val.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, folder=None, dataset=None, transforms=None, mode='train'):
        super(CocoDetection, self).__init__()
        anno_file = f"{dataset}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, f"{folder}", 'images', 'rgb')
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(
            self.img_root
        )
        self.anno_path = os.path.join(root, f"{folder}", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(
            self.anno_path
        )

        self.mode = mode
        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        self.pixel_mean = np.array([0.485, 0.456, 0.406])
        self.pixel_std = np.array([0.229, 0.224, 0.225])

        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90
        # 将缺失的类别名称设置成N/A
        coco_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if mode == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("coco91_indices.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if mode == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(
        self, img_id: int, coco_targets: list, w: int = None, h: int = None
    ):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        rgb_loc = os.path.join(self.img_root, coco.loadImgs(img_id)[0]['file_name'])
        sar_loc = rgb_loc.replace("rgb", "sar")
        dsm_loc = rgb_loc.replace("rgb", "dsm")

        rgb = load_rgb(rgb_loc)
        sar = load_sar(sar_loc)
        dsm = load_dsm(dsm_loc)

        #rgb = resiz_4pl(rgb, (224, 224))
        #sar = resiz_4pl(sar, (224, 224))
        #dsm = resiz_4pl(dsm, (224, 224))

        c, w, h = rgb.shape
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            rgb = torch.from_numpy(rgb)
            sar = torch.from_numpy(sar)
            dsm = torch.from_numpy(dsm)

        return rgb, sar, dsm, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))




if __name__ == '__main__':
    train = CocoDetection("/media/yusin/Elements/DFC2023", dataset="track2")
    print(len(train))
    img, tgt = train[0]
    print(tgt['masks'].shape)

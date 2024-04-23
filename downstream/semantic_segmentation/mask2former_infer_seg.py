import numpy as np
import os
import cv2
import glob
import tqdm
import torch
from torch.nn import functional as F
from modeling.MaskFormerModel_vit import MaskFormerModel
from utils.misc import load_parallal_model
from dataset.multimodal_quadruplet import load_quadruplet, resiz_4pl, RandomCrop, Index2Color, Color2Index
import matplotlib.pyplot as plt
import rasterio

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def denormalize_s2(imgs):
    for i in range(3):
        imgs[i, :, :] = imgs[i, :, :] * S2_STD[i] + S2_MEAN[i]
    return imgs

def normalization(data):
    if torch.is_tensor(data):
        _range = torch.max(data) - torch.min(data)
        return (data - torch.min(data)) / _range
    else:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.
    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.
    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.
    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


class Segmentation(MaskFormerModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.device = torch.device("cuda", cfg.local_rank)

        # data processing program
        self.padding_constant = 2 ** 5  # resnet 总共下采样5次
        self.test_dir = cfg.TEST.TEST_DIR
        self.output_dir = cfg.TEST.SAVE_DIR
        self.imgMaxSize = cfg.INPUT.CROP.MAX_SIZE
        self.pixel_mean = np.array(cfg.DATASETS.PIXEL_MEAN)
        self.pixel_std = np.array(cfg.DATASETS.PIXEL_STD)
        self.model = MaskFormerModel(cfg, pretrained=None, frozen_stages=None)

        pretrain_weights = cfg.MODEL.PRETRAINED_WEIGHTS
        assert os.path.exists(pretrain_weights), f'please check weights file: {cfg.MODEL.PRETRAINED_WEIGHTS}'
        self.load_model(pretrain_weights)

        # Inference
        self.panoptic_on = cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
        self.instance_on = cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        self.semantic_on = cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        self.sem_seg_postprocess_before_inference = cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE
        self.test_topk_per_image = cfg.MODEL.MASK_FORMER.TEST.TEST_TOPK_PER_IMAGE

        self.crop = RandomCrop(256, 1)

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    def load_model(self, pretrain_weights):
        state_dict = torch.load(pretrain_weights, map_location='cuda:0')

        ckpt_dict = state_dict['model']
        self.last_lr = state_dict['lr']
        self.start_epoch = state_dict['epoch']
        self.model = load_parallal_model(self.model, ckpt_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("loaded pretrain mode:{}".format(pretrain_weights))

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio

    def resize_padding(self, img, outsize):
        c, w, h = img.shape
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)
        ow, oh = round(w * ratio), round(h * ratio)
        img = resiz_4pl(img, (ow, oh))
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = np.pad(img, ((0, 0), (top, bottom), (left, right)), 'constant')  # 左 顶 右 底 顺时针
        return img, [left, top, right, bottom]

    def image_preprocess(self, s1, s2, dem, dnw):
        img_height, img_width = s1.shape[1], s1.shape[2]
        this_scale = self.imgMaxSize / max(img_height, img_width)
        target_width = img_width * this_scale
        target_height = img_height * this_scale
        input_width = int(self.round2nearest_multiple(target_width, self.padding_constant))
        input_height = int(self.round2nearest_multiple(target_height, self.padding_constant))

        s11, padding_info = self.resize_padding(s1, (input_width, input_height))
        #s21, _ = self.resize_padding(s2, (input_width, input_height))
        #dem1, _ = self.resize_padding(dem, (input_width, input_height))
        #dnw1, _ = self.resize_padding(dnw[np.newaxis, :, :], (input_width, input_height))
        #dnw1 = dnw1.squeeze()

        transformer_info = {'padding_info': padding_info, 'scale': this_scale,
                            'input_size': (input_height, input_width)}

        s1 = torch.from_numpy(s1).float().unsqueeze(0).to(self.device)
        s2 = torch.from_numpy(s2).float().unsqueeze(0).to(self.device)
        dem = torch.from_numpy(dem).float().unsqueeze(0).to(self.device)
        dnw = torch.from_numpy(dnw).long().unsqueeze(0).to(self.device)

        input_tensor = {'s1': s1, 's2': s2, 'dem': dem, 'dnw': dnw}

        return input_tensor, transformer_info

    def postprocess(self, pred_mask, transformer_info, target_size):
        oh, ow = pred_mask.shape[0], pred_mask.shape[1]
        padding_info = transformer_info['padding_info']

        left, top, right, bottom = padding_info[0], padding_info[1], padding_info[2], padding_info[3]
        mask = pred_mask[top: oh - bottom, left: ow - right]
        mask = cv2.resize(mask.astype(np.uint8), dsize=target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    @torch.no_grad()
    def forward(self, inputs):
        s1, s2, dem, dnw = inputs['s1'], inputs['s2'], inputs['dem'], inputs['dnw']
        img_height, img_width = s1.shape[1], s1.shape[2]
        input_tensor, transformer_info = self.image_preprocess(s1, s2, dem, dnw)
        b, c, input_height, input_width = input_tensor['s2'].shape

        outputs = self.model(input_tensor)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )

        input_sizes = [(input_height, input_width), ]
        processed_results = []

        for mask_cls_result, mask_pred_result, input_size in zip(mask_cls_results, mask_pred_results, input_sizes):

            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, input_size, img_height, img_width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = self.semantic_inference(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = sem_seg_postprocess(r.to(torch.float), input_size, img_height, img_width)
                processed_results[-1]["sem_seg"] = r

        return processed_results

    @torch.no_grad()
    def forward_segmentation(self, s2_list=None):
        if s2_list is None or len(s2_list) == 0:
            s2_list = glob.glob(self.test_dir + '/s2/*.tif')
        for s2_path in tqdm.tqdm(s2_list):
            img_name = os.path.basename(s2_path)
            seg_name = img_name[:-4] + '_seg.png'
            output_path = os.path.join(self.output_dir, seg_name)

            s1_path = s2_path.replace("_s2_", "_s1_").replace("s2", "s1")
            dem_path = s2_path.replace("_s2_", "_dem_").replace("s2", "dem")
            dnw_path = s2_path.replace("_s2_", "_dnw_").replace("s2", "dnw")
            alt_path = s2_path.replace("_s2_", "_alt_").replace("s2", "alt")

            #with rasterio.open(alt_path) as data:
            #    real = data.read([1, 2, 3])

            input_dict = {"s1": s1_path, "s2": s2_path, "dem": dem_path, "dnw": dnw_path, "lc": alt_path, "id": img_name}
            inputs = load_quadruplet(input_dict, True, True, True, True, unlabeled=False)
            inputs = self.crop(inputs, unlabeled=False)
            img = normalization(denormalize_s2(inputs['s2'][[2, 1, 0], :, :]))
            target = inputs['label']
            result = self.forward(inputs)
            sem_seg = result[-1]["sem_seg"].to('cpu')
            #print(target.shape)
            sem_seg = torch.argmax(sem_seg, dim=0) + 1 # plus one just because ignore the zero class but plot from zero !!!!!!!!!!!!!!!!!!!
            semantic_result = Index2Color(sem_seg)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(np.rollaxis(img, 0, 3) * 3)
            ax2.imshow(semantic_result.astype(np.int))
            ax3.imshow(Index2Color(target).astype(np.int))
            #ax2.imshow(Index2Color(Color2Index(real).squeeze()).astype(np.int))  # cannot be float, has to be int
            #ax3.imshow(np.rollaxis(real, 0, 3))
            for ax in [ax1, ax2, ax3]:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig(output_path)
            plt.close()

    @torch.no_grad()
    def semantic_inference(self, mask_cls, mask_pred):
        #mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # wrong , but don't know why
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # here ignore one class, that is zeros !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

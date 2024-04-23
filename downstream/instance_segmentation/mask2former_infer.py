from statistics import mode
from fvcore.common.config import CfgNode
import numpy as np
import os
import cv2
import glob
import tqdm
from PIL import Image
from PIL import ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from modeling.MaskFormerModel import MaskFormerModel
from utils.misc import load_parallal_model
from utils.misc import ADEVisualize
from utils import Instances, Boxes, filter_instances_with_score, Visualizer, ColorMode
from dataset.register_ade20k_panoptic import get_metadata
from dataset.register_ade20k_full import _get_ade20k_full_meta
from utils.catalog import MetadataCatalog


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
        self.visualize = ADEVisualize()
        self.model = MaskFormerModel(cfg)

        pretrain_weights = cfg.MODEL.PRETRAINED_WEIGHTS
        assert os.path.exists(pretrain_weights), f'please check weights file: {cfg.MODEL.PRETRAINED_WEIGHTS}'
        self.load_model(pretrain_weights)

        # Inference
        self.metadata = get_metadata()
        self.panoptic_on = cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
        self.instance_on = cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        self.semantic_on = cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        self.sem_seg_postprocess_before_inference = cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE
        self.test_topk_per_image = cfg.MODEL.MASK_FORMER.TEST.TEST_TOPK_PER_IMAGE

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

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = (img - self.pixel_mean) / self.pixel_std
        img = img.transpose((2, 0, 1))
        return img

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

    def resize_padding(self, img, outsize, Interpolation=Image.BILINEAR):
        w, h = img.size
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)
        ow, oh = round(w * ratio), round(h * ratio)
        img = img.resize((ow, oh), Interpolation)
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
        return img, [left, top, right, bottom]

    def image_preprocess(self, img):
        img_height, img_width = img.shape[0], img.shape[1]
        this_scale = self.imgMaxSize / max(img_height, img_width)
        target_width = img_width * this_scale
        target_height = img_height * this_scale
        input_width = int(self.round2nearest_multiple(target_width, self.padding_constant))
        input_height = int(self.round2nearest_multiple(target_height, self.padding_constant))

        img, padding_info = self.resize_padding(Image.fromarray(img), (input_width, input_height))
        img = self.img_transform(img)

        transformer_info = {'padding_info': padding_info, 'scale': this_scale,
                            'input_size': (input_height, input_width)}
        input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        return input_tensor, transformer_info

    def postprocess(self, pred_mask, transformer_info, target_size):
        oh, ow = pred_mask.shape[0], pred_mask.shape[1]
        padding_info = transformer_info['padding_info']

        left, top, right, bottom = padding_info[0], padding_info[1], padding_info[2], padding_info[3]
        mask = pred_mask[top: oh - bottom, left: ow - right]
        mask = cv2.resize(mask.astype(np.uint8), dsize=target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    @torch.no_grad()
    def forward(self, img):
        img_height, img_width = img.size[1], img.size[0]
        input_tensor, transformer_info = self.image_preprocess(np.array(img))
        b, c, input_height, input_width = input_tensor.shape

        outputs = self.model(input_tensor)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(input_tensor.shape[-2], input_tensor.shape[-1]),
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

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                instance_r = self.instance_inference(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    @torch.no_grad()
    def forward_instance_segmentation(self, img_list=None):
        if img_list is None or len(img_list) == 0:
            #img_list = glob.glob(self.test_dir + '/*.[jp][pn]g')
            img_list = glob.glob(self.test_dir + '/*.tif')
        for image_path in tqdm.tqdm(img_list):
            img_name = os.path.basename(image_path)
            seg_name = img_name.split('.')[0] + '_seg.png'
            output_path = os.path.join(self.output_dir, seg_name)
            img = Image.open(image_path).convert('RGB')
            result = self.forward(img)
            instance = result[0]["instances"].to('cpu')
            #v = Visualizer(np.array(img), self.metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            v = Visualizer(np.array(img), None, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            semantic_result = v.draw_instance_predictions(predictions=instance).get_image()
            cv2.imwrite(output_path, semantic_result)

    @torch.no_grad()
    def forward_segmentation(self, img_list=None):
        #print(dict2obj(self.metadata).stuff_colors)
        if img_list is None or len(img_list) == 0:
            img_list = glob.glob(self.test_dir + '/*.[jp][pn]g')
        for image_path in tqdm.tqdm(img_list):
            img_name = os.path.basename(image_path)
            seg_name = img_name.split('.')[0] + '_seg.png'
            output_path = os.path.join(self.output_dir, seg_name)
            img = Image.open(image_path).convert('RGB')
            result = self.forward(img)
            sem_seg = result[-1]["sem_seg"].to('cpu')
            sem_seg = torch.argmax(sem_seg, dim=0)
            v = Visualizer(np.array(img), dict2obj(self.metadata), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            semantic_result = v.draw_sem_seg(sem_seg).get_image()
            cv2.imwrite(output_path, semantic_result)


    @torch.no_grad()
    def forward_panoptic_segmentation(self, img_list=None):
        if img_list is None or len(img_list) == 0:
            img_list = glob.glob(self.test_dir + '/*.[jp][pn]g')
        for image_path in tqdm.tqdm(img_list):
            img_name = os.path.basename(image_path)
            seg_name = img_name.split('.')[0] + '_seg.png'
            output_path = os.path.join(self.output_dir, seg_name)
            img = Image.open(image_path).convert('RGB')
            result = self.forward(img)
            panoptic_seg = result[0]["panoptic_seg"].to('cpu')
            v = Visualizer(np.array(img), self.metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            semantic_result = v.draw_panoptic_seg(panoptic_seg).get_image()  ##########not completed#################
            cv2.imwrite(output_path, semantic_result)

    @torch.no_grad()
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                       device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    @torch.no_grad()
    def semantic_inference(self, mask_cls, mask_pred):
        #mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # wrong , but don't know why
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    @torch.no_grad()
    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    @torch.no_grad()
    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        #scores = F.softmax(mask_cls, dim=-1)[:, :-1] # wrong, don't know why
        scores = F.softmax(mask_cls, dim=-1)[:, 1:]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result




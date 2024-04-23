import numpy as np
import os
import cv2
import glob
import tqdm
from PIL import Image
import torch
from torch.nn import functional as F
from modeling.MaskFormerModel_vit import MaskFormerModel
from utils.misc import load_parallal_model
from utils.misc import ADEVisualize
from utils import Instances, Boxes, filter_instances_with_score, Visualizer, ColorMode, COCOeval
from dataset.register_ade20k_panoptic import get_metadata
from dataset.my_json_dataset_resize import load_dsm, load_rgb, load_sar, resiz_4pl
from pycocotools import mask as maskUtils
import json
from utils.utils import filter_instances_with_area, remove_overlap
from utils import Instances, Boxes, filter_instances_with_score, Visualizer, ColorMode, BitMasks


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        elif isinstance(obj, bytes):
            return obj.decode('utf-8')

        return json.JSONEncoder.default(self, obj)

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

def build_coco_results(image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    * boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
    * labels (Int64Tensor[N]): the predicted labels for each image
    * scores (Tensor[N]): the scores or each prediction
    * masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
    """
    # If no results, return an empty list
    if rois is None:
        return []
    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i] + 1
            score = scores[i]
            bbox = np.around(rois[i, :], 1)
            mask = masks[i, :, :]
            result = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


class Segmentation2json(MaskFormerModel):
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
        self.model = MaskFormerModel(cfg, pretrained=None, frozen_stages=None)

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


    @torch.no_grad()
    def forward_instance_segmentation(self, eval_loder, save_name):
        processed_results = []
        for (rgb, sar, dsm, targets) in eval_loder:
            rgb = list(image.float().to(self.device) for image in rgb)
            sar = list(image.float().to(self.device) for image in sar)
            dsm = list(image.float().to(self.device) for image in dsm)
            targets = [{k: v for k, v in t.items()} for t in targets]
            img_id = [i['image_id'] for i in targets]
            img_id = [element.item() for lis in img_id for element in lis]

            rgb = torch.stack(rgb, dim=0)
            sar = torch.stack(sar, dim=0)
            dsm = torch.stack(dsm, dim=0)
            inputs = {'s1': sar, 's2': rgb, 'dem': dsm}

            outputs = self.model(inputs)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(rgb.shape[-2], rgb.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            _, _, img_height, img_width = rgb.shape
            input_sizes = [(img_height, img_width), ]


            for mask_cls_result, mask_pred_result, input_size in zip(mask_cls_results, mask_pred_results, input_sizes):


                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, input_size, img_height, img_width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # instance segmentation inference
                if self.instance_on:
                    instance_r = self.instance_inference(mask_cls_result, mask_pred_result)
                    #instance_r = filter_instances_with_area(instance_r.to('cpu'), 0.0040)
                    #instance_r = remove_overlap(instance_r, 0.2)
                    # Convert results to COCO format
                    image_results = build_coco_results(img_id, instance_r.pred_boxes.cpu().numpy(), instance_r.pred_classes.cpu().numpy(),
                                                       instance_r.scores.cpu().numpy(), instance_r.pred_masks.cpu().numpy().astype(np.uint8))
                    #print("Number of detections : ", len(instance_r.pred_boxes.cpu().numpy()))
                    #print("Classes Predicted : ", instance_r.pred_classes.cpu().numpy())
                    #print("Scores : ", instance_r.scores.cpu().numpy())
                    processed_results.extend(image_results)

        with open(save_name + '.json', "w") as f:
            json.dump(processed_results, f, cls=NumpyEncoder)
        print('writing finished!')


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
                    "img_id": targets_per_image.gt_image_id,
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
        #result.pred_boxes = torch.zeros(mask_pred.size(0), 4)
        result.pred_boxes = torch.tensor([0, 0, 255, 255]).unsqueeze(0).repeat(mask_pred.size(0), 1)
        # Uncomment the following to get boxes from masks (this is slow)
        #result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        #print(result.pred_boxes)
        #result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_tensor()


        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result




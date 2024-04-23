#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MaskFormerModel.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于DeformTransAtten的分割网络
'''

# here put the import lib
from torch import nn
from addict import Dict

from .multimae.multimae_big_imcomplete import ViTMAE
from .pixel_decoder.msdeformattn_vit import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class MaskFormerHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.pixel_decoder = self.pixel_decoder_init(cfg, input_shape)
        self.predictor = self.predictor_init(cfg)

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        mask_classification = True
        predictor = MultiScaleMaskedTransformerDecoder(in_channels,
                                                       num_classes,
                                                       mask_classification,
                                                       hidden_dim,
                                                       num_queries,
                                                       nheads,
                                                       dim_feedforward,
                                                       dec_layers,
                                                       pre_norm,
                                                       mask_dim,
                                                       enforce_input_project)
        return predictor

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions


class MaskFormerModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.backbone_feature_shape = dict()
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape)

    def build_backbone(self, cfg, *args, **kwargs):

        backbone = ViTMAE(cfg, *args, **kwargs)
        channels = [backbone.dim_tokens, backbone.dim_tokens, backbone.dim_tokens, backbone.dim_tokens]

        for i, channel in enumerate(channels):
            self.backbone_feature_shape[f'res{i + 2}'] = Dict({'channel': channel, 'stride': 2**(i+3)})
        return backbone

    def forward(self, inputs):
        feat_lel = {}
        features = self.backbone(inputs)
        feat_lel['res2'] = features[0]
        feat_lel['res3'] = features[1]
        feat_lel['res4'] = features[2]
        feat_lel['res5'] = features[3]
        outputs = self.sem_seg_head(feat_lel)
        return outputs

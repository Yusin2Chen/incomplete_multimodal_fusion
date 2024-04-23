# Copyright (c) EPFL VILAB. All rights reserved.
import itertools
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union, Tuple
import torch
from einops import rearrange, repeat, pack, unpack
from torch import nn
from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F
from .multimae_utils import trunc_normal_
from torch.nn.init import normal_
from .zorro_utils import Block, Attention, TokenTypes, LayerNorm, exists, Mlp
from .input_adapters import PatchedInputAdapter, FusionInputAdapter
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

import sys
sys.path.append("..")
from pixel_decoder.ops.modules import MSDeformAttn


__all__ = [
    'pretrain_multimae_base',
    'pretrain_multimae_large',
    'ViTMAE',
]


class MultiMAE(nn.Module):
    """MultiMAE: Multi-task Multi-modal Masked Autoencoder
    This module performs masking in its forward pass.
    The MultiViT module defined below inherits from this module and performs a regular forward pass,
    and should be used instead for downstream tasks


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """

    def __init__(self,
                 input_adapters: Dict[str, nn.Module],
                 output_adapters: Optional[Dict[str, nn.Module]],
                 dim_tokens: int = 768,
                 depth: int = 12,
                 dim_head: int = 64,
                 heads: int = 8,
                 ff_mult: int = 4,
                 num_fusion_tokens: int = 16,
                 return_token_types: Tuple[TokenTypes] = (
                 TokenTypes.S1, TokenTypes.S2, TokenTypes.DEM, TokenTypes.FUSION),
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = LayerNorm):
        super().__init__()

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        # copy fusion token in num_patches times added to return token
        assert num_fusion_tokens == input_adapters['s1'].num_patches
        # Initialize parameters
        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)

        self.return_tokens = nn.Parameter(torch.randn(1, self.max_return_tokens, dim_tokens))  # return tokens
        trunc_normal_(self.return_tokens, std=0.02)
        self.attn_pool = Attention(dim=dim_tokens, dim_head=dim_head, heads=heads)
        self.fusion_tokens = nn.Parameter(torch.randn(1, num_fusion_tokens, dim_tokens))  # fusion tokens
        trunc_normal_(self.fusion_tokens, std=0.02)

        mlp_hidden_dim = int(dim_tokens * 4.0)
        self.mlp = Mlp(in_features=dim_tokens, hidden_features=mlp_hidden_dim)

        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=dim_tokens, dim_head=dim_head, heads=heads, ff_mult=ff_mult, drop_path=dpr[i],
                  norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = LayerNorm(dim_tokens)
        self.dim_tokens = dim_tokens
        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens'}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'input_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for task, adapter in self.output_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'output_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def forward_features(self,
                x: Union[Dict[str, torch.Tensor], torch.Tensor],
                mask_inputs: bool = False,
                task_masks: Dict[str, torch.Tensor] = None,
                num_encoded_tokens: int = 128):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.

        :param x: Input tensor or dictionary of tensors
        :param mask_inputs: Set to True to enable random masking of input patches
        :param task_masks: Optional dictionary of task->mask pairs.
        :param num_encoded_tokens: Number of tokens to randomly select for encoder.
            Only used if mask_inputs is True.
        """
        ## Processing input modalities
        # If input x is a Tensor, assume it's RGB
        x = {'s1': x} if isinstance(x, torch.Tensor) else x
        batch, device = x['s1'].shape[0], x['s1'].device
        # Need image size for tokens->image reconstruction
        # We assume that at least one of rgb or semseg is given as input before masking
        if 's1' in x:
            B, C, H, W = x['s1'].shape
        else:
            B, C, H, W = list(x.values())[0].shape  # TODO: Deal with case where not all have same shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        # Construct fusion tokens
        fusion_tokens = repeat(self.fusion_tokens, '() n d -> b n d', b=B)
        fusion_tokens, N_H, N_W = self.input_adapters['fusion'](fusion_tokens)

        # Select random subset of tokens from the chosen input tasks and concatenate them
        if mask_inputs:
            num_encoded_tokens = num_encoded_tokens if num_encoded_tokens is not None else self.num_encoded_tokens
        else:
            num_encoded_tokens = sum([tensor.shape[1] for tensor in input_task_tokens.values()])

        s1_tokens = input_task_tokens['s1']
        s2_tokens = input_task_tokens['s2']
        dem_tokens = input_task_tokens['dem']
        # construct all tokens
        s1_tokens, s2_tokens, dem_tokens, fusion_tokens = \
            map(lambda t: rearrange(t, 'b ... d -> b (...) d'),
                (s1_tokens, s2_tokens, dem_tokens, fusion_tokens))
        # -----------------------mask zorro----------------------------
        tokens = torch.cat((s1_tokens, s2_tokens, dem_tokens, fusion_tokens), 1)
        # construct mask (thus zorro)
        token_types = torch.tensor(list((
            *((TokenTypes.S1.value,) * s1_tokens.shape[-2]),
            *((TokenTypes.S2.value,) * s2_tokens.shape[-2]),
            *((TokenTypes.DEM.value,) * dem_tokens.shape[-2]),
            *((TokenTypes.FUSION.value,) * fusion_tokens.shape[-2]),
        )), device=device, dtype=torch.long)

        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')

        # the logic goes
        # every modality, including fusion can attend to self
        zorro_mask = token_types_attend_from == token_types_attend_to
        # fusion can attend to everything
        tokens_join = token_types_attend_from == (
                torch.ones_like(token_types_attend_to, device=device, dtype=torch.long) * TokenTypes.FUSION.value)
        zorro_mask = zorro_mask.long() | tokens_join

        # Transformer forward pass
        for blk in self.blocks:
            tokens = blk(tokens, zorro_mask.bool())
        tokens = self.norm(tokens)
        # -------------------------Output features-------------------------------------------------------
        # Decode tokens (encoded fusion tokens) for each task using task-specific output adapters
        encoder_fusion_tokens = tokens[:, num_encoded_tokens:, :]

        return encoder_fusion_tokens


class ViTAdapter(MultiMAE):
    def __init__(self, pretrain_size=224, num_heads=8, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False, *args, **kwargs):

        super().__init__(heads=num_heads, *args, **kwargs)

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.dim_tokens

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, (2, 2), (2, 2))

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False


DOMAIN_CONF = {
    's1': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    },
    's2': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
    },
    'dem': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    },
    'fusion': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(FusionInputAdapter, num_channels=1),
    }
}

def ViTMAE(args):
    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.MultiMAE.patch_size,
            image_size=args.MultiMAE.input_size,
        )
        for domain in args.MultiMAE.in_domains
    }

    # Add input adapter for fusion tokens
    if args.MultiMAE.extra_fusion_token:
        input_adapters['fusion'] = DOMAIN_CONF['fusion']['input_adapter'](
            stride_level=DOMAIN_CONF['fusion']['stride_level'],
            patch_size_full=args.MultiMAE.patch_size,
            image_size=args.MultiMAE.input_size,
        )

    model = ViTBaseline(
        input_adapters=input_adapters,
        output_adapters=None,
        num_fusion_tokens=196,  # number of fusion tokens
        return_token_types=(TokenTypes.S1, TokenTypes.S2, TokenTypes.DEM, TokenTypes.FUSION),
        drop_path_rate=args.MultiMAE.drop_path
    )
    return model


def pretrain_multimae_tiny(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=384,
        depth=12,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_layer=LayerNorm,
        **kwargs
    )
    return model


def pretrain_multimae_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_layer=LayerNorm,
        **kwargs
    )
    return model


def pretrain_multimae_large(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_layer=LayerNorm,
        **kwargs
    )
    return model


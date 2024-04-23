# Copyright (c) EPFL VILAB. All rights reserved.
import itertools
import math
import os.path
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union, Tuple
import torch
from einops import rearrange, repeat, pack, unpack
from torch import nn
from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F
from .multimae_utils import trunc_normal_
from .zorro_utils import Block, Attention, TokenTypes, LayerNorm, exists, Mlp
from .input_adapters import PatchedInputAdapter, FusionInputAdapter


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
        num_out_tokens = input_adapters['s1'].num_patches
        return_token_types = list(return_token_types)
        return_token_types = return_token_types + [return_token_types[-1]] * num_out_tokens
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

        # final attention pooling - each modality pool token can only attend to its own tokens
        return_tokens = self.return_tokens
        return_token_types_tensor = self.return_token_types_tensor
        return_tokens = repeat(return_tokens, '() n d -> b n d', b=batch)
        pos_return_tokens, N_H, N_W = self.input_adapters['fusion'](return_tokens[:, 4:, :])
        return_tokens = torch.cat((return_tokens[:, 0:4, :], pos_return_tokens), dim=1)
        # ------------------------------------mapping links----------------------------------------------
        return_tokens_attend_from = rearrange(return_token_types_tensor, 'i -> i 1')
        pool_mask = return_tokens_attend_from == token_types_attend_to
        pool_join = return_tokens_attend_from == (
            torch.ones_like(token_types_attend_to, device=device, dtype=torch.long) * TokenTypes.FUSION.value)
        pool_mask = pool_mask | pool_join
        # -------------------------Output features-------------------------------------------------------
        # Decode tokens (encoded fusion tokens) for each task using task-specific output adapters
        return_tokens = self.attn_pool(return_tokens, context=tokens, attn_mask=pool_mask)
        return_tokens = return_tokens + self.mlp(self.norm(return_tokens))
        encoder_fusion_tokens = return_tokens[:, 4:, :]

        return encoder_fusion_tokens

class ViTBaseline(MultiMAE):
    def __init__(self, pretrained=None, pretrain_size=224, frozen_stages=12, freeze_attn=False, freeze_ffn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_stages = frozen_stages
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [i for i in range(-1, self.num_block, self.num_block // 4)][1:]

        embed_dim = self.dim_tokens

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(embed_dim, embed_dim, (2, 2), (2, 2)),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, (2, 2), (2, 2))
        ])
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, (2, 2), (2, 2))
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

        if os.path.exists(pretrained):
            self.init_weights(pretrained=pretrained)
            if self.frozen_stages is not None:
                self._freeze_stages()  # important
                print('backbone is freezed!')


    def init_weights(self, pretrained: str = None):
        if isinstance(pretrained, str):
            ckpt = torch.load(pretrained)
            pretrained_dict = ckpt['model']
            self.load_state_dict(pretrained_dict, strict=False)

    def load_state_dict(self, state_dict, strict=False, logger=None):
        """Load state_dict to a module.
        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.
        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
            logger (:obj:`logging.Logger`, optional): Logger to log the error
                message. If not specified, print function will be used.
        """
        unexpected_keys = []
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} and '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, own_state[name].size(), param.size()))
        missing_keys = set(own_state.keys()) - set(state_dict.keys())

        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}\n'.format(
                ', '.join(unexpected_keys)))
        if missing_keys:
            err_msg.append('missing keys in source state_dict: {}\n'.format(
                ', '.join(missing_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            elif logger is not None:
                logger.warn(err_msg)
            else:
                print(err_msg)


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

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward_features(self,
                x: Union[Dict[str, torch.Tensor], torch.Tensor],
                mask_inputs: bool = False,
                task_masks: Dict[str, torch.Tensor] = None,
                num_encoded_tokens: int = 128):

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
        # final attention pooling - each modality pool token can only attend to its own tokens
        return_tokens = self.return_tokens
        return_token_types_tensor = self.return_token_types_tensor
        return_tokens = repeat(return_tokens, '() n d -> b n d', b=batch)
        pos_return_tokens, N_H, N_W = self.input_adapters['fusion'](return_tokens[:, 4:, :])
        return_tokens = torch.cat((return_tokens[:, 0:4, :], pos_return_tokens), dim=1)
        # ------------------------------------mapping links----------------------------------------------
        return_tokens_attend_from = rearrange(return_token_types_tensor, 'i -> i 1')
        pool_mask = return_tokens_attend_from == token_types_attend_to
        pool_join = return_tokens_attend_from == (
                torch.ones_like(token_types_attend_to, device=device, dtype=torch.long) * TokenTypes.FUSION.value)
        pool_mask = pool_mask | pool_join

        outs = []
        # Transformer forward pass
        for index, blk in enumerate(self.blocks):
            tokens = blk(tokens, zorro_mask.bool())
        return_tokens = self.attn_pool(return_tokens, context=tokens, attn_mask=pool_mask)
        return_tokens = return_tokens + self.mlp(self.norm(return_tokens))
        outs.append(return_tokens[:, 4:, :])
        outs.append(return_tokens[:, 4:, :])
        outs.append(return_tokens[:, 4:, :])
        outs.append(return_tokens[:, 4:, :])
        return outs, N_H, N_W

    def forward(self, input_dict):
        outs, H, W = self.forward_features(input_dict)
        f1, f2, f3, f4 = outs
        bs, n, dim = f1.shape
        f1 = self.norm(f1).transpose(1, 2).reshape(bs, dim, H, W)
        f2 = self.norm(f2).transpose(1, 2).reshape(bs, dim, H, W)
        f3 = self.norm(f3).transpose(1, 2).reshape(bs, dim, H, W)
        f4 = self.norm(f4).transpose(1, 2).reshape(bs, dim, H, W)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()

        return [f1, f2, f3, f4]

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.input_adapters.eval()
            for param in self.input_adapters.parameters():
                param.requires_grad = False
            self.return_tokens.requires_grad = False
            self.fusion_tokens.requires_grad = False
            for param in self.attn_pool.parameters():
                param.requires_grad = False
            for param in self.mlp.parameters():
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
                m.norm2.eval()
                m.mlp.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False
                for param in m.mlp.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.input_adapters.eval()
            for param in self.input_adapters.parameters():
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

def ViTMAE(args, *argss, **kwargs):
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
        num_fusion_tokens=16,  # number of fusion tokens
        return_token_types=(TokenTypes.S1, TokenTypes.S2, TokenTypes.DEM, TokenTypes.FUSION),
        drop_path_rate=args.MultiMAE.drop_path,
        dim_tokens=384,
        depth=12,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_layer=LayerNorm,
        frozen_stages=11,
        pretrained=args.MODEL.BACKBONE.PRETRAINED_WEIGHTS,
        *argss, **kwargs
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


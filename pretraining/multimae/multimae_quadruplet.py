# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import itertools
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union, Tuple

import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributions.dirichlet import Dirichlet

from .multimae_utils import trunc_normal_
from .zorro_utils_quadruplet import Block, Attention, TokenTypes, LayerNorm, exists, Mlp

__all__ = [
    'pretrain_multimae_base',
    'pretrain_multimae_large',
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
                 num_global_tokens: int = 1,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 dim_head: int = 64,
                 heads: int = 8,
                 ff_mult: int = 4,
                 num_fusion_tokens: int = 16,
                 return_token_types: Tuple[TokenTypes] = (TokenTypes.S1, TokenTypes.S2, TokenTypes.DEM, TokenTypes.FUSION),
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
        #num_out_tokens = input_adapters['s1'].num_patches
        #return_token_types = list(return_token_types)
        #return_token_types = return_token_types[:-1] + [return_token_types[-1]] * num_out_tokens
        #return_token_types = return_token_types+ [return_token_types[-1]] * num_out_tokens
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
        # Additional learnable tokens that can be used by encoder to process/store global information
        #self.num_global_tokens = num_global_tokens
        #self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        #trunc_normal_(self.global_tokens, std=0.02)

        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=dim_tokens, dim_head=dim_head, heads=heads, ff_mult=ff_mult, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = LayerNorm(dim_tokens)

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

    def sample_alphas(self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:])
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        print(rand_per_sample_choice.shape)
        alphas_tensor = torch.index_select(valid_task_choices, 0, rand_per_sample_choice)
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks(self,
                            input_tokens: Dict[str, torch.Tensor],
                            num_encoded_tokens: int,
                            alphas: Union[float, List[float]] = 1.0,
                            sample_tasks_uniformly: bool = False):
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select
        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        B = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device

        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            # 测试时这里B一定要改1
            #alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)  # 不同
            alphas = self.sample_alphas(1, len(input_tokens), alphas=alphas)  # 改成相同
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            #task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)  # 不同
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((1,)).to(device)  # 改成相同

        samples_per_task = (task_sampling_dist * num_encoded_tokens).round().long()

        task_masks = []
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in input_tokens.values()]
        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            #noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]   # 不同
            noise = torch.rand(1, num_tokens, device=device)  # noise in [0, 1]  # 改成相同
            ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            #mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)   # 不同
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(1, -1)  # 改成相同
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)

        #print(task_masks)
        mask_all = torch.cat(task_masks, dim=1)

        # --------人为增加万一不够固定数目的token--------------
        #samples = torch.sum(mask_all[0, :] * (-1) + 1)
        #if samples < num_encoded_tokens:
        #    #print('low num', samples, mask_all)
        #    ids_shuffle = torch.argsort(mask_all, dim=1)  # 按顺序
        #    #print(ids_shuffle)
        #    #print(ids_shuffle[:, samples: num_encoded_tokens])
        #    mask_all[:, ids_shuffle[:, samples: num_encoded_tokens]] = 0

        #print(mask_all)
        #ids_shuffle = torch.argsort(mask_all, dim=1)  # 按顺序
        #print(ids_shuffle)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)  # 随机排列
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {domain: mask.repeat(B, 1) for domain, mask in zip(input_tokens.keys(), task_masks)}  # 在batch上扩展相同
        #print(task_masks)
        return task_masks, ids_keep.repeat(B, 1), ids_restore.repeat(B, 1)  # 在batch上扩展相同

    @staticmethod
    def make_mask(N_H, N_W, xy_idxs, full_tasks=[], indicate_visible=True, flatten=True, device='cuda'):
        """
        Creates masks for each task, given lists of un-masked x,y coordinates.
        """
        xy_idxs = {
            k: torch.LongTensor(v)
            for k, v in xy_idxs.items()
        }

        task_masks = {
            k: torch.ones(N_H, N_W).to(device)
            for k in xy_idxs.keys()
        }

        for k in xy_idxs.keys():
            if len(xy_idxs[k]) > 0:
                task_masks[k][xy_idxs[k][:, 1], xy_idxs[k][:, 0]] = 0

        for task in full_tasks:
            task_masks[task][:] = 0

        if not indicate_visible:
            task_masks = {k: 1 - v for k, v in task_masks.items()}

        if flatten:
            task_masks = {k: v.flatten().unsqueeze(0) for k, v in task_masks.items()}

        return task_masks

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                'num_tokens': num_tokens,
                'has_2d_posemb': True,  # TODO: Modify when adding non-2D tasks
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            i += num_tokens
            input_info['tasks'][domain] = d

        input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        #input_info['num_global_tokens'] = self.num_global_tokens

        return input_info

    def forward(self, 
                x: Union[Dict[str, torch.Tensor], torch.Tensor], 
                mask_inputs: bool = True,
                task_masks: Dict[str, torch.Tensor] = None,
                num_encoded_tokens: int = 128,
                alphas: Union[float, List[float]] = 1.0,
                sample_tasks_uniformly: bool = False,
                fp32_output_adapters: List[str] = [],
                return_token_indices: Optional[Tuple[int]] = None):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.

        :param x: Input tensor or dictionary of tensors
        :param mask_inputs: Set to True to enable random masking of input patches
        :param task_masks: Optional dictionary of task->mask pairs.
        :param num_encoded_tokens: Number of tokens to randomly select for encoder.
            Only used if mask_inputs is True.
        :param alphas: Dirichlet distribution parameter alpha for task sampling.
            Higher alpha = harder, less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True if tasks should be uniformly presampled,
            before Dirichlet sampling decides share of masked tokens between them.
        :param fp32_output_adapters: List of task identifiers to force output adapters to
            run with mixed precision turned off for stability reasons.
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
        fusion_tokens = self.input_adapters['fusion'](fusion_tokens)

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, image_size=(H, W))

        # Select random subset of tokens from the chosen input tasks and concatenate them
        if mask_inputs:
            num_encoded_tokens = num_encoded_tokens if num_encoded_tokens is not None else self.num_encoded_tokens
        else:
            num_encoded_tokens = sum([tensor.shape[1] for tensor in input_task_tokens.values()])

        ## Generating masks
        if task_masks is None:
            task_masks, ids_keep, ids_restore = self.generate_random_masks(
                input_task_tokens,
                num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly
            )
        else:
            mask_all = torch.cat([task_masks[task] for task in input_task_tokens.keys()], dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :(mask_all == 0).sum()]

        s1_idx = ((task_masks['s1'][0, :] == 0).nonzero(as_tuple=True)[0])
        s1_tokens = input_task_tokens['s1'][:, s1_idx, :]
        s2_idx = ((task_masks['s2'][0, :] == 0).nonzero(as_tuple=True)[0])
        s2_tokens = input_task_tokens['s2'][:, s2_idx, :]
        dem_idx = ((task_masks['dem'][0, :] == 0).nonzero(as_tuple=True)[0])
        dem_tokens = input_task_tokens['dem'][:, dem_idx, :]
        dnw_idx = ((task_masks['dnw'][0, :] == 0).nonzero(as_tuple=True)[0])
        dnw_tokens = input_task_tokens['dnw'][:, dnw_idx, :]
        #fusion_tokens = fusion_tokens[:, torch.cat((s1_idx, s2_idx, dem_idx), 0), :]
        #print(torch.cat((s1_idx, s2_idx, dem_idx), 0), fusion_tokens.shape)
        #print(s1_tokens.shape)
        #print(input_task_tokens['s1'].shape)
        #print(task_masks['s1'])

        #input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)
        # Apply mask
        #input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))
        #print(input_tokens.shape)
        # construct all tokens
        s1_tokens, s2_tokens, dem_tokens, dnw_tokens, fusion_tokens = \
            map(lambda t: rearrange(t, 'b ... d -> b (...) d'),
                (s1_tokens, s2_tokens, dem_tokens, dnw_tokens, fusion_tokens))

        # Add global tokens to input tokens
        #global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        #input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        ## Transformer forward pass
        #encoder_tokens = self.encoder(input_tokens)

        # -----------------------mask zorro----------------------------
        tokens = torch.cat((s1_tokens, s2_tokens, dem_tokens, dnw_tokens, fusion_tokens), 1)
        #tokens2, ps = pack((s1_tokens, s2_tokens, dem_tokens, fusion_tokens), 'b * d')
        # construct mask (thus zorro)
        token_types = torch.tensor(list((
            *((TokenTypes.S1.value,) * s1_tokens.shape[-2]),
            *((TokenTypes.S2.value,) * s2_tokens.shape[-2]),
            *((TokenTypes.DEM.value,) * dem_tokens.shape[-2]),
            *((TokenTypes.DNW.value,) * dnw_tokens.shape[-2]),
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

        if exists(return_token_indices):
            assert len(set(return_token_indices)) == len(return_token_indices), 'all indices must be unique'
            assert all([indice < self.max_return_tokens for indice in return_token_indices]), 'indices must range from 0 to max_num_return_tokens - 1'
            return_token_indices = torch.tensor(return_token_indices, dtype=torch.long, device=device)
            return_token_types_tensor = return_token_types_tensor[return_token_indices]
            return_tokens = return_tokens[return_token_indices]

        return_tokens = repeat(return_tokens, '() n d -> b n d', b=batch)
        #return_tokens = torch.cat((return_tokens[:, 0:4, :], self.input_adapters['fusion'](return_tokens[:, 4:, :])), dim=1)
        # ----------------------------------------------------------------
        return_tokens_attend_from = rearrange(return_token_types_tensor, 'i -> i 1')
        pool_mask = return_tokens_attend_from == token_types_attend_to
        pool_join = return_tokens_attend_from == (
                torch.ones_like(token_types_attend_to, device=device, dtype=torch.long) * TokenTypes.FUSION.value)
        pool_mask = pool_mask | pool_join
        # ----------------------------------------------------------------
        ori_tokens = tokens[:, :num_encoded_tokens, :]
        return_tokens = self.attn_pool(return_tokens, context=tokens, attn_mask=pool_mask)
        return_tokens = return_tokens + self.mlp(self.norm(return_tokens))

        # -------------------------Output decoders-------------------------------------------------------
        if self.output_adapters is None:
            return tokens, return_tokens, task_masks

        # Decode tokens (encoded fusion tokens) for each task using task-specific output adapters
        encoder_fusion_tokens = tokens[:, num_encoded_tokens:, :]

        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_fusion_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.output_adapters
            if domain not in fp32_output_adapters
        }
        # Force running selected output adapters in fp32 mode
        with torch.cuda.amp.autocast(enabled=False):
            for domain in fp32_output_adapters:
                if domain not in self.output_adapters:
                    continue
                preds[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_fusion_tokens.float(),
                    input_info=input_info,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )

        
        return preds, task_masks, return_tokens, ori_tokens, encoder_fusion_tokens


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


import argparse
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from einops import rearrange

import utils
from multimae.multimae_crossattn import pretrain_multimae_base, pretrain_multimae_tiny
from multimae.zorro_utils import TokenTypes as T
from multimae.input_adapters import PatchedInputAdapter, FusionInputAdapter
from multimae.output_adapters_simple import SpatialOutputAdapter
from utils.multimodal_dfc2023 import RandomCrop, load_rgb_sar_dsm

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import colors
import copy


DOMAIN_CONF = {
    's1': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    },
    's2': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
    },
    'dem': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    },
    'fusion': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(FusionInputAdapter, num_channels=1),
    }
}

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser('MultiMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (default: %(default)s)')
    # Task parameters
    parser.add_argument('--in_domains', default='s1-s2-dem', type=str, help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='s1-s2-dem', type=str, help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--extra_fusion_token', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--model', default='pretrain_multimae_base', type=str, metavar='MODEL', help='Name of model to train (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens', default=98, type=int, help='Number of tokens to randomly choose for encoder (default: %(default)s)')
    parser.add_argument('--num_global_tokens', default=1, type=int, help='Number of global tokens to add to encoder (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int, help='Base patch size for image-like modalities (default: %(default)s)')
    parser.add_argument('--input_size', default=256, type=int, help='Images input size for backbone (default: %(default)s)')
    parser.add_argument('--alphas', type=float, default=1.0, help='Dirichlet alphas concentration parameter (default: %(default)s)')
    parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true', help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')

    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true', help='Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true', help='Set to True/False to enable/disable decoder cross attention.')
    parser.add_argument('--decoder_dim', default=256, type=int, help='Token dimension inside the decoder layers (default: %(default)s)')
    parser.add_argument('--decoder_depth', default=2, type=int, help='Number of self-attention layers after the initial cross attention (default: %(default)s)')
    parser.add_argument('--decoder_num_heads', default=8, type=int, help='Number of attention heads in decoder (default: %(default)s)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: %(default)s)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/workspace/DFC2023', type=str, help='dataset path')
    # Misc.
    parser.add_argument('--output_dir', default='', help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='Device to use for training / testing')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


def get_model(args):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            image_size=args.input_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )
        for domain in args.out_domains
    }

    # Add input adapter for fusion tokens
    if args.extra_fusion_token:
        input_adapters['fusion'] = DOMAIN_CONF['fusion']['input_adapter'](
            stride_level=DOMAIN_CONF['fusion']['stride_level'],
            patch_size_full=args.patch_size,
            image_size=args.input_size,
        )

    model = pretrain_multimae_tiny(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        num_fusion_tokens=256,  # number of fusion tokens
        return_token_types=(T.S1, T.S2, T.DEM, T.FUSION),
        drop_path_rate=args.drop_path
    )

    # load pre-trained model
    print('==> loading pre-trained model')
    ckpt = torch.load('./save_attention/checkpoint-1339.pth') #799
    model.load_state_dict(ckpt['model'], strict=True)

    return model


def normalization(data):
    if torch.is_tensor(data):
        _range = torch.max(data) - torch.min(data)
        new_data = (torch.clone(data) - torch.min(data)) / _range
        return new_data
    else:
        _range = np.max(data) - np.min(data)
        new_data = (copy.deepcopy(data) - torch.min(data)) / _range
        return new_data

def normalization2(data):
    if torch.is_tensor(data):
        _range = torch.max(data) - torch.min(data)
        new_data = (torch.clone(data) - torch.min(data)) / _range
        return new_data
    else:
        _range = np.max(data) - np.min(data)
        new_data = (copy.deepcopy(data) - torch.min(data)) / _range
        return new_data

def get_masked_image(img, mask, image_size=256, patch_size=16, mask_value=0.0):
    img_token = rearrange(
        img.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    img_token[mask.detach().cpu() != 0] = mask_value
    img = rearrange(
        img_token,
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    return img

def get_pred_with_input(gt, pred, mask, image_size=256, patch_size=16):
    gt_token = rearrange(
        gt.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    pred_token = rearrange(
        pred.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    pred_token[mask.detach().cpu() == 0] = gt_token[mask.detach().cpu() == 0]
    img = rearrange(
        pred_token,
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    return img


# standar
rgb_MEAN = np.array([81.29692, 87.93711, 72.041306])
rgb_STD  = np.array([39.61512, 35.407978, 35.84708])

def denormalize_s2(imgs):
    for i in range(3):
        imgs[:, i, :, :] = imgs[:, i, :, :] * rgb_STD[i] + rgb_MEAN[i]
    return imgs


sar_MEAN = np.array([-7.9447875, ])
sar_STD  = np.array([2.777256, ])

def denormalize_s1(imgs):
    for i in range(1):
        imgs[:, i, :, :] = imgs[:, i, :, :] * sar_STD[i] + sar_MEAN[i]
    return imgs


dem_MEAN = np.array([5.0160093, ])
dem_STD  = np.array([7.6128364, ])

def denormalize_dem(imgs):
    for i in range(1):
        imgs[:, i, :, :] = imgs[:, i, :, :] * dem_STD[i] + dem_MEAN[i]
    return imgs

def plot_predictions(input_dict, preds, masks, image_size=256):
    masked_rgb = get_masked_image(
        normalization(denormalize_s2(torch.clone(input_dict['s2']))),
        masks['s2'],
        image_size=image_size,
        mask_value=1.0
    )[0].permute(1, 2, 0).detach().cpu()
    masked_sar = get_masked_image(
        normalization2(denormalize_s1(torch.clone(input_dict['s1']))),
        masks['s1'],
        image_size=image_size,
        mask_value=np.nan
    )[0].permute(1, 2, 0).detach().cpu()
    masked_dem = get_masked_image(
        normalization(torch.clone(input_dict['dem'])),
        masks['dem'],
        image_size=image_size,
        mask_value=np.nan
    )[0].permute(1, 2, 0).detach().cpu()

    pred_rgb = normalization(denormalize_s2(preds['s2']))[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()
    pred_sar = normalization(denormalize_s1(preds['s1']))[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()
    pred_dem = normalization(preds['dem'])[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0)

    grid[0].imshow(masked_rgb)
    grid[1].imshow(pred_rgb)
    grid[2].imshow(normalization(denormalize_s2(torch.clone(input_dict['s2'])))[0].permute(1, 2, 0).detach().cpu())

    #vmin = np.nanmin(masked_sar.numpy())
    #vmax = np.nanmax(masked_sar.numpy())
    #norm = colors.Normalize(vmin=0, vmax=1)
    grid[3].imshow(masked_sar, vmin=0, vmax=1.0)
    grid[4].imshow(pred_sar)
    grid[5].imshow(normalization(denormalize_s1(torch.clone(input_dict['s1'])))[0].permute(1, 2, 0).detach().cpu())

    grid[6].imshow(masked_dem)
    grid[7].imshow(pred_dem)
    grid[8].imshow(normalization(torch.clone(input_dict['dem']))[0].permute(1, 2, 0).detach().cpu())

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    fontsize = 16
    grid[0].set_title('Masked inputs', fontsize=fontsize)
    grid[1].set_title('MultiMAE predictions', fontsize=fontsize)
    grid[2].set_title('Original Reference', fontsize=fontsize)
    grid[0].set_ylabel('RGB', fontsize=fontsize)
    grid[3].set_ylabel('SAR', fontsize=fontsize)
    grid[6].set_ylabel('DEM', fontsize=fontsize)
    plt.savefig('output.jpg')




def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)
    model = model.to(device).eval()

    # Load image and display pseudo labels
    data = {"rgb": '/workspace/DFC2023/track2/images/rgb/GF2_Brasilia_-15.8652_-47.9337.tif',
            "sar": '/workspace/DFC2023/track2/images/sar/GF2_Brasilia_-15.8652_-47.9337.tif',
            "dsm": '/workspace/DFC2023/track2/images/dsm/GF2_Brasilia_-15.8652_-47.9337.tif', 'id': 'test'}
    # Center crop and resize RGB
    data = load_rgb_sar_dsm(data, True, True, True, unlabeled=True)
    crop_data = data
    # Plot loaded RGB image and sar and dem
    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    #ax[0].imshow(denormalize_s2(np.rollaxis(crop_data['s2'], 0, 3)))
    #ax[0].set_title('RGB', fontsize=16)
    #ax[1].imshow(denormalize_s1(np.rollaxis(crop_data['s1'], 0, 3)))
    #ax[1].set_title('SAR', fontsize=16)
    #ax[2].imshow(denormalize_dem(np.rollaxis(crop_data['dem'], 0, 3)))
    #ax[2].set_title('DEM', fontsize=16)
    #for a in ax:
    #    a.set_xticks([])
    #    a.set_yticks([])
    #plt.show()
    #plt.savefig('input.jpg')

    # Pre-process RGB, depth and semseg to the MultiMAE input format
    input_dict = crop_data
    del input_dict['id']
    # To GPU
    input_dict = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in input_dict.items()}

    # infer
    torch.manual_seed(1)  # change seed to resample new mask
    num_encoded_tokens = 256  # the number of visible tokens
    alphas = 1.0  # Dirichlet concentration parameter
    with torch.no_grad():
        preds, masks, _, _, _, _, _, _ = model(
            input_dict,
            num_encoded_tokens=num_encoded_tokens,
            alphas=alphas,
        )
    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    plot_predictions(input_dict, preds, masks)

    '''
    # modify the sampled mask and use it as input
    # 1 - masked patch  0 - non-masked
    masks['s1'].fill_(1)
    masks['dem'].fill_(1)
    task_masks = {k: v.to(device) for k, v in masks.items()}

    preds, masks, _ = model.forward(
        input_dict,
        mask_inputs=True,
        task_masks=task_masks
    )

    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    plot_predictions(input_dict, preds, masks)
    '''



if __name__ == '__main__':
    opts = get_args()
    main(opts)
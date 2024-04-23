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
from multimae.multimae_quadruplet import pretrain_multimae_base, pretrain_multimae_tiny
from multimae.zorro_utils_quadruplet import TokenTypes as T
from multimae.input_adapters import PatchedInputAdapter, FusionInputAdapter, SemSegInputAdapter
from multimae.output_adapters_simple import SpatialOutputAdapter
from utils.multimodal_quadruplet import RandomCrop, load_quadruplet, Index2Color

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import colors


colors_list = ['#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635', '#DFC35A', '#C4281B', '#A59B8F', '#B39FE1',
                           'linen', 'black', 'cyan', 'pink', 'purple', 'navy', 'silver']
bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
cmap = colors.ListedColormap(colors_list)
norm = colors.BoundaryNorm(bounds, cmap.N)

DOMAIN_CONF = {
    's1': {
        'channels': 2,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=2),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=2),
    },
    's2': {
        'channels': 4,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=4),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=4),
    },
    'dem': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    },
    'dnw': {
        'num_classes': 9,
        'stride_level': 1,
        'input_adapter': partial(SemSegInputAdapter, num_classes=9,
                                 dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=9),
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
    parser.add_argument('--in_domains', default='s1-s2-dem-dnw', type=str, help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='s1-s2-dem-dnw', type=str, help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--extra_fusion_token', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--model', default='pretrain_multimae_base', type=str, metavar='MODEL', help='Name of model to train (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens', default=256, type=int, help='Number of tokens to randomly choose for encoder (default: %(default)s)')
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
        return_token_types=(T.S1, T.S2, T.DEM, T.DNW, T.FUSION),
        drop_path_rate=args.drop_path
    )

    # load pre-trained model
    print('==> loading pre-trained model')
    ckpt = torch.load('./save/checkpoint-64.pth')
    model.load_state_dict(ckpt['model'], strict=True)

    return model


def normalization(data):
    if torch.is_tensor(data):
        _range = torch.max(data) - torch.min(data)
        return (data - torch.min(data)) / _range
    else:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

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
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def denormalize_s2(imgs):
    for i in range(3):
        imgs[:, i, :, :] = imgs[:, i, :, :] * S2_STD[i] + S2_MEAN[i]
    return imgs


S1_MEAN = np.array([-9.020017, -15.73008])
S1_STD  = np.array([3.5793820, 3.671725])

def denormalize_s1(imgs):
    for i in range(1):
        imgs[:, i, :, :] = imgs[:, i, :, :] * S1_STD[i] + S1_MEAN[i]
    return imgs



def plot_predictions(input_dict, preds, masks, image_size=256):
    masked_rgb = get_masked_image(
        normalization(denormalize_s2(input_dict['s2'][:, [2, 1, 0], :, :])),
        masks['s2'],
        image_size=image_size,
        mask_value=1.0
    )[0].permute(1, 2, 0).detach().cpu()
    masked_sar = get_masked_image(
        normalization(denormalize_s1(input_dict['s1'][:, [0], :, :])),
        masks['s1'],
        image_size=image_size,
        mask_value=np.nan
    )[0].permute(1, 2, 0).detach().cpu()
    masked_dem = get_masked_image(
        normalization(input_dict['dem']),
        masks['dem'],
        image_size=image_size,
        mask_value=np.nan
    )[0].permute(1, 2, 0).detach().cpu()
    masked_dnw = get_masked_image(
        input_dict['dnw'].float().unsqueeze(1),
        masks['dnw'],
        image_size=image_size,
        mask_value=np.nan
    )[0].permute(1, 2, 0).detach().cpu()

    pred_rgb = normalization(denormalize_s2(preds['s2'][:, [2, 1, 0], :, :]))[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()
    pred_sar = normalization(denormalize_s1(preds['s1'][:, [0], :, :]))[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()
    pred_dem = normalization(preds['dem'])[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()
    pred_dnw = preds['dnw'].argmax(1).unsqueeze(1)[0].permute(1, 2, 0).clamp(0, 1).detach().cpu()

    fig = plt.figure(figsize=(13, 17))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 3), axes_pad=0)

    grid[0].imshow(masked_rgb)
    grid[1].imshow(pred_rgb)
    grid[2].imshow(normalization(denormalize_s2(input_dict['s2'][:, [2, 1, 0], :, :]))[0].permute(1, 2, 0).detach().cpu())

    grid[3].imshow(masked_sar)
    grid[4].imshow(pred_sar)
    grid[5].imshow(normalization(denormalize_s1(input_dict['s1'][:, [0], :, :]))[0].permute(1, 2, 0).detach().cpu())

    grid[6].imshow(masked_dem)
    grid[7].imshow(pred_dem)
    grid[8].imshow(normalization(input_dict['dem'])[0].permute(1, 2, 0).detach().cpu())

    grid[9].imshow(masked_dnw, cmap=cmap, norm=norm)
    grid[10].imshow(pred_dnw, cmap=cmap, norm=norm)
    grid[11].imshow(input_dict['dnw'].unsqueeze(1)[0].permute(1, 2, 0).detach().cpu(), cmap=cmap, norm=norm)

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
    grid[9].set_ylabel('DNW', fontsize=fontsize)
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
    data = {"s1": '/workspace/s1s2dsm_patch/a1/s1_0/a1_s1_0_0.tif',
            "s2": '/workspace/s1s2dsm_patch/a1/s2_0/a1_s2_0_0.tif',
            "dem": '/workspace/s1s2dsm_patch/a1/dem_0/a1_dem_0_0.tif',
            "dnw": '/workspace/s1s2dsm_patch/a1/dnw_0/a1_dnw_0_0.tif', 'id': 'test'}
    # Center crop and resize RGB
    crop = RandomCrop(256)
    data = load_quadruplet(data, True, True, True, True, unlabeled=True)
    data = crop(data)
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
    #input_dict = crop_data
    input_dict = data
    del input_dict['id']
    # To GPU
    input_dict = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in input_dict.items()}

    # infer
    torch.manual_seed(1)  # change seed to resample new mask
    num_encoded_tokens = 256  # the number of visible tokens
    alphas = 1.0  # Dirichlet concentration parameter
    with torch.no_grad():
        preds, masks, _, _, _ = model(
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
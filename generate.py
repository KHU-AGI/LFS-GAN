import torch
import os

import numpy as np
import torch.nn.functional as F

from argparse import ArgumentParser
from lifelong_model import LifelongGenerator as Generator
from PIL import Image
from torchvision import transforms, utils
from torch import nn
from tqdm import tqdm

def generate(model, result_path, n_samples, args, mask=None):
    with torch.no_grad():
        model.eval()
        os.makedirs(result_path, exist_ok=True)
        for i in tqdm(range(n_samples), desc='Generating...'):
            z = torch.randn(1, 512).cuda()
            fake_x, _ = model([z], truncation=args.truncation, truncation_latent=None)  
            utils.save_image(fake_x, os.path.join(result_path, '{}.png'.format(i)), normalize=True, value_range=(-1, 1))


if __name__=='__main__':
    argparser=ArgumentParser()
    argparser.add_argument('--pretrained_ckpt', type=str, default="checkpoints/ffhq.pt")
    argparser.add_argument('--ckpt', type=str, required=True)
    argparser.add_argument('--result_path', type=str, required=True)
    argparser.add_argument('--n_samples', type=int, default=5000)
    argparser.add_argument('--truncation', type=int,default=1)
    argparser.add_argument('--truncation_mean', type=int, default=4096)
    argparser.add_argument('--size', type=int, default=256)
    argparser.add_argument('--latent', type=int, default=512)
    argparser.add_argument('--n_mlp', type=int, default=8)
    argparser.add_argument('--channel_multiplier', type=int, default=2)
    argparser.add_argument('--subspace_std', type=float, default=0.1)
    argparser.add_argument('--noise', type=str, default=None)
    argparser.add_argument('--rank', type=int, default=1)
    argparser.add_argument("--left_use_act", action="store_true")
    argparser.add_argument("--left_use_add", action="store_true")
    args = argparser.parse_args()
    torch.manual_seed(10)
    
    pretrained_ckpt = torch.load(args.pretrained_ckpt)
    ckpt = torch.load(args.ckpt)

    model = Generator(args.size, args.latent, args.n_mlp, lifelong=True, rank=args.rank, left_use_act=args.left_use_act, left_use_add=args.left_use_add).cuda()
    model.load_state_dict(pretrained_ckpt['g_ema'], strict=False)

    state_dict = model.state_dict()

    state_dict.update(ckpt['g_ema'])

    model.load_state_dict(state_dict, strict=False)

    generate(model, args.result_path, args.n_samples, args)
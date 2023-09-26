import argparse
import random
import torch
import os
import sys
import lpips

import torch.nn as nn
import numpy as np

from torchvision import utils, transforms
from tqdm import tqdm
from PIL import Image
from dataset import MultiResolutionDataset
from torch.utils.data import DataLoader

def intra_cluster_dist(args):
    torch.set_grad_enabled(False)
    lpips_fn = lpips.LPIPS(net='vgg').to(args.device)
    lpips_fn.eval()

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    real_dataset = MultiResolutionDataset(path=args.real_path, transform=transform, resolution=256)
    real_loader = DataLoader(real_dataset, batch_size=10, shuffle=False)
    data_size = len(real_dataset)
    print(data_size)
    real_tensors = None
    for real_tensor in real_loader:
        if real_tensors is None:
            real_tensors = real_tensor.to(args.device)
        else:
            real_tensors = torch.cat((real_tensors, real_tensor.to(args.device)), dim=0)
    fake_images = os.listdir(args.fake_path)[:1000]
    cluster = {}
    for i in range(data_size):
        cluster[i] = []
    # clustering
    pbar = tqdm(fake_images, desc='Clustering...')
    for fake_image_path in pbar:
        image_path = os.path.join(args.fake_path, fake_image_path)
        fake_image = Image.open(image_path)
        fake_tensor = transform(fake_image).to(args.device)
        dists = np.zeros(data_size)
        for i in range(data_size):
            out = lpips_fn(fake_tensor, real_tensors[i]).view(-1).item()
            dists[i] = out
        closest_cluster = np.argmin(dists)
        cluster[closest_cluster].append(os.path.join(args.fake_path, fake_image_path))

    # compute average pairwise distance
    print('Clustered as :', [len(cluster[c]) for c in cluster])
    cluster_length_list = [len(cluster[c]) for c in cluster]
    p = torch.tensor(cluster_length_list) / len(fake_images) + 1e-5
    entropy = -1 * (p * p.log10())
    print(f'Entropy : {entropy.sum().item():.4f}')
    dists = []
    cluster_size = args.cluster_size
    cluster = {c: cluster[c][:cluster_size] for c in cluster}
    total_length = sum([len(cluster[c]) * (len(cluster[c]) - 1) for c in cluster]) // 2
    with tqdm(range(total_length), desc='Computing...') as pbar:
        for c in cluster:
            temp = []
            cluster_length = len(cluster[c])
            for i in range(cluster_length):
                img1 = Image.open(cluster[c][i])
                img1 = transform(img1).cuda()
                for j in range(i + 1, cluster_length):
                    img2 = Image.open(cluster[c][j])
                    img2 = transform(img2).cuda()
                    pairwise_dist = lpips_fn(img1, img2)
                    temp.append(pairwise_dist.item())
                    pbar.update(1)
            dists.append(np.mean(temp))
    dists = np.array(dists)
    print(f'LPIPS of each cluster:')
    for i in range(data_size):
        print(f'Cluster {i} : {dists[i]:.4f}')
    
    entropy = entropy.numpy()
    new_metric = 0
    for i in range(data_size):
        if np.isnan(dists[i]):
            continue
        else:
            new_metric += entropy[i] * dists[i]
    print(f'Intra-Cluster LPIPS : {dists[~np.isnan(dists)].mean():.4f}')
    print(f'Balanced Intra-cluster LPIPS : {new_metric:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', type=str, required=True) # processed_data/Sketches/10shot/0
    parser.add_argument('--fake_path', type=str, required=True) # fake_images/LFS/Sketches_Sketches_5000
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cluster_size', type=int, default=50)
    args = parser.parse_args()
    intra_cluster_dist(args)
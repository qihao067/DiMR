"""This file contains code for extracting imagenet features.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/baofff/U-ViT/blob/main/scripts/extract_imagenet_feature.py
"""

import torch.nn as nn
import numpy as np
import torch
from datasets import ImageNet
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm
torch.manual_seed(0)
np.random.seed(0)


def main(resolution=256):

    dataset = ImageNet(path='path/to/imagenet', resolution=resolution, random_flip=False)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    idx = 0
    for batch in tqdm(train_dataset_loader):
        img, label = batch
        img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()

        label = torch.cat([label, label], dim=0)
        label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            np.save(f'imagenet{resolution}_features/{idx}.npy', (moment, lb))
            idx += 1

    print(f'save {idx} files')


if __name__ == "__main__":
    main()

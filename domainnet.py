# -*- coding: utf-8 -*-
"""
Code from: https://github.com/virajprabhu/SENTRY
"""
import os
import sys
import copy
import random
import numpy as np
import torch
from torchvision import transforms
import utils

from torch.utils.data import Dataset
import os
from PIL import Image
import pdb

VALID_DOMAINS = [
    'clipart',
    'infograph',
    'painting',
    'quickdraw',
    'real',
    'sketch'
]

SENTRY_DOMAINS = [
    'clipart',
    'painting',
    'real',
    'sketch'
]

NUM_CLASSES_DICT = {
    'full': 345,
    'sentry': 40
}

VALID_SPLITS = ['train', 'test']

VALID_VERSIONS = ['full', 'sentry']

ROOT = '/p/lustre1/trivedi1/vision_data/DomainNet'
SENTRY_SPLITS_ROOT = '/p/lustre1/trivedi1/vision_data/SENTRY_SPLITS' 


def load_dataset(domains, split, version):
    if len(domains) == 1 and domains[0] == 'all':
        if version == 'sentry':
            domains = SENTRY_DOMAINS
        else:
            domains = VALID_DOMAINS

    data = []
    for domain in domains:
        if version == 'sentry':
            idx_file = os.path.join(SENTRY_SPLITS_ROOT, f'{domain}_{split}_mini.txt')
        else:
            idx_file = os.path.join(ROOT, f'{domain}_{split}.txt')
        with open(idx_file, 'r') as f:
            data += [line.split() for line in f]
    return data


class DomainNet(Dataset):
    def __init__(self, domain, split='train', root=ROOT,
                 transform=None, unlabeled=False, verbose=False,
                 version='sentry'):
        super().__init__()

        if version not in VALID_VERSIONS:
            raise ValueError(f'dataset version must be in {VALID_VERSIONS} but was {version}')
        domain_list = domain.split(',')
        for domain in domain_list:
            if domain != 'all' and version == 'full' and domain not in VALID_DOMAINS:
                raise ValueError(f'domain must be in {VALID_DOMAINS} but was {domain}')
            if domain != 'all' and version == 'sentry' and domain not in SENTRY_DOMAINS:
                raise ValueError(f'domain must be in {SENTRY_DOMAINS} but was {domain}')
        if split not in VALID_SPLITS:
            raise ValueError(f'split must be in {VALID_SPLITS} but was {split}')
        self._root_data_dir = root
        self._domain_list = domain_list
        self._split = split
        self._transform = transform
        self._version = version

        self._unlabeled = unlabeled
        self.data = load_dataset(domain_list, split, version)
        if verbose:
            print(f'Loaded domains {", ".join(domain_list)}, split is {split}')
            print(f'Total number of images: {len(self.data)}')
            print(f'Total number of classes: {self.get_num_classes()}')
            print("transform: ",self._transform)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, y = self.data[idx]
        x = Image.open(os.path.join(self._root_data_dir, path))
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        # if self._unlabeled:
        #     return x, -1
        # else:
        # print(x.shape)
        return x, int(y)

    def get_num_classes(self):
        return len(set([self.data[idx][1] for idx in range(len(self.data))]))

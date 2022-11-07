from random import shuffle
# from torch_cka import CKA
import os
import time
from breeds import Breeds
import torch
import timm
import tqdm 
import argparse
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from utils import get_transform, NUM_CLASSES_DICT, test,norm_dict 
import pdb 
import random
import domainnet 
import clip_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from blended_cifar_mnist import blendedCIFARMNIST, blendedSTLMNIST

class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """
        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            # print("Named Module: ",name)
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches
                assert not torch.isnan(self.hsic_matrix).any(), pdb.set_trace()
        # pdb.set_trace()
        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }



def arg_parser():
    parser = argparse.ArgumentParser(
    description='Extract CKA',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_1',
        type=str,
        default='domainnet-sketch',
        choices=['cifar10','stl10','STL10','domainnet-sketch','domainnet-real','living17','blendedCIFAR','blendedSTL'])
    
    parser.add_argument(
        '--dataset_2',
        type=str,
        default='domainnet-sketch',
        choices=['stl10','none','domainnet-sketch','domainnet-painting','domainnet-real','domainnet-clipart'])
    
    parser.add_argument(
        '--arch',
        type=str,
        default='clip-RN50',
        choices=['resnet50','clip-RN50'])
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )
    
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=128
    )    
    parser.add_argument(
        '--model_1_ckpt',
        type=str,
    )
    
    parser.add_argument(
        '--model_2_ckpt',
        type=str,
    )
    parser.add_argument(
        '--dataset_1_trainaug',
        type=str,
        default='test', #intentional for CKA!
        choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test']
    )
    
    parser.add_argument(
        '--dataset_1_testaug',
        type=str,
        default='test',
        choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test']
    )
    parser.add_argument(
        '--dataset_2_trainaug',
        type=str,
        default='test', #intentional for CKA!
        choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test']
    )
    
    parser.add_argument(
        '--dataset_2_testaug',
        type=str,
        default='test',
        choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test']
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=1
    ) 
    
    parser.add_argument(
        '--save_name',
        type=str,
    )
    parser.add_argument(
        "--correlation_strength",
        type=float,
    ) 
    args = parser.parse_args()
    return args  

def get_oodloader(args,dataset):
    if "clip" in args.arch:
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073],[0.26862954, 0.26130258, 0.27577711])
        print("OOD is using CLIP Noralizer")
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),normalize])
    if dataset.upper() == 'STL10':

        ood_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
            split='test',
            download=False,
            transform=transform)

        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        ood_dataset.labels = stl_to_cifar_indices[ood_dataset.labels]
        ood_dataset = torch.utils.data.Subset(ood_dataset,np.where(ood_dataset.labels != -1)[0])
    elif "domainnet" in dataset.lower():
        domain_name = dataset.split("-")[-1]
        ood_dataset = domainnet.DomainNet(domain=domain_name, 
            split='test',
            root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet/",
            transform=transform,
            verbose=False) 
    else:
        print("ERROR ERROR ERROR")
        print("Not Implemented Yet. Exiting")
        exit()
    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=wif) 
    return ood_loader


def load_moco_ckpt(model,pretrained_ckpt):
    ckpt = torch.load(pretrained_ckpt)
    new_keys = [(k, k.replace("module.encoder_q.","")) for k in list(ckpt['state_dict'].keys())]
    for old_k, new_k in new_keys:
        ckpt['state_dict'][new_k] = ckpt['state_dict'].pop(old_k)
    incompatible, unexpected = model.load_state_dict(ckpt['state_dict'],strict=False)
    print("Incompatible Keys: ", incompatible)
    print("Unexpected Keys: ",unexpected)
    return model

#TODO: move to utils. Manually stopping shuffling and dropping last here.
def get_dataloaders(args,dataset,train_aug, test_aug, train_transform,test_transform):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    if train_aug not in ['mixup','cutmix','cutout']:
        if dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=train_transform)
        elif dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=train_transform)
        elif dataset == 'STL10' or dataset == 'stl10':
            train_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
                split='train',
                download=False,
                transform=train_transform)

            stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
            train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
            train_dataset = torch.utils.data.Subset(train_dataset,np.where(train_dataset.labels != -1)[0])
        elif "domainnet" in dataset.lower():
            domain_name = dataset.split("-")[-1]
            train_dataset = domainnet.DomainNet(domain=domain_name, 
                split='train',
                root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet",
                transform=train_transform) 
        elif "living17" in dataset.lower():
            train_dataset= Breeds(root='/usr/workspace/trivedi1/vision_data/ImageNet', 
                breeds_name='living17', 
                info_dir='/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified',
                source=True, 
                target=False, 
                split='train', 
                transform=train_transform) 
        else:
            print("***** ERROR ERROR ERROR ******")
            print("=> Invalid Dataset Selected, Exiting")
            exit()
    else:
        #augmentation will be applied in training loop!
        if "clip" in args.arch:
            print("Using Clip Normalization!") 
            normalize = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                transforms.Normalize(norm_dict['clip_mean'], norm_dict["clip_std"]),
                ])
        else:
            normalize = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                transforms.Normalize(norm_dict[dataset+'_mean'], norm_dict[dataset + "_std"]),
                ])
        if dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=normalize)
        elif dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=normalize)
        elif dataset == 'STL10' or dataset == 'stl10':
            train_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
                split='train',
                download=False,
                transform=normalize)

            stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
            train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
            train_dataset = torch.utils.data.Subset(train_dataset,np.where(train_dataset.labels != -1)[0])
        elif "domainnet" in dataset.lower():
            domain_name = dataset.split("-")[-1]
            train_dataset = domainnet.DomainNet(domain=domain_name, 
                split='train',
                root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet",
                transform=normalize) 
        elif "living17" in args.dataset.lower():
            train_dataset= Breeds(root='/usr/workspace/trivedi1/vision_data/ImageNet', 
                breeds_name='living17', 
                info_dir='/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified',
                source=True, 
                target=False, 
                split='train', 
                transform=normalize) 
    """
    Create Test Dataloaders.
    """
    if dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=test_transform)
        NUM_CLASSES=10
    elif dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=test_transform)
        NUM_CLASSES=100
    elif dataset == 'STL10' or dataset == 'stl10':
            test_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
                split='test',
                download=False,
                transform=test_transform)

            stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
            test_dataset.labels = stl_to_cifar_indices[test_dataset.labels]
            test_dataset = torch.utils.data.Subset(test_dataset,np.where(test_dataset.labels != -1)[0])
    elif "domainnet" in dataset.lower():
        domain_name = dataset.split("-")[-1]
        test_dataset = domainnet.DomainNet(domain=domain_name, 
            split='test',
            root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet",
            transform=test_transform) 
        NUM_CLASSES=40
    elif "living17" in dataset.lower():
        test_dataset = Breeds(root='/usr/workspace/trivedi1/vision_data/ImageNet', 
            breeds_name='living17', 
            info_dir='/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified',
            source=True, 
            target=False, 
            split='val', 
            transform=test_transform) 
    
    else:
        print("***** ERROR ERROR ERROR ******")
        print("xxx Invalid Dataset Selected, Exiting")
        exit()
     
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g)


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g)

    return train_loader, test_loader

def main():
    args = arg_parser() 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("args.dataset_1",args.dataset_1) 
    """
    Create Model 1
    """
 
    if "clip" in args.model_1_ckpt and "scratch" not in args.model_1_ckpt:
        encoder_type = args.arch.split("-")[-1]
        print("\t => Clip Encoder: ",encoder_type)
        net_1 = clip_model.ClipModel(model_name=encoder_type,scratch=True)
        net_1.reset_classifier(NUM_CLASSES_DICT[args.dataset_1])
        net_1 = torch.nn.DataParallel(net_1,device_ids=[0]).cuda()
        ckpt = torch.load(args.model_1_ckpt)
        incomp, unexpected = net_1.load_state_dict(ckpt['state_dict'])
        print("LOADED M1 CKPT")
        print("Incompatible Keys: ",incomp)
        print("Unexpected Keys: ",unexpected) 
        m1_name = args.model_1_ckpt 
   
    elif "resnet50" in args.model_1_ckpt and "scratch" not in args.model_1_ckpt:
        net_1 = timm.create_model(args.arch,pretrained=False)
        net_1.reset_classifier(NUM_CLASSES_DICT[args.dataset_1])
        net_1 = torch.nn.DataParallel(net_1,device_ids=[0]).cuda()
        ckpt = torch.load(args.model_1_ckpt)
        incomp,unexpected = net_1.load_state_dict(ckpt['state_dict'])
        print("LOADED M1 CKPT")
        print("Incompatible Keys: ",incomp)
        print("Unexpected Keys: ",unexpected) 
        m1_name = args.model_1_ckpt 
    elif "scratch-clip" in args.model_1_ckpt: #load a image-net pretrained clip only.
        encoder_type = args.arch.split("-")[-1]
        print("\t => Clip Encoder: ",encoder_type)
        net_1 = clip_model.ClipModel(model_name=encoder_type,scratch=False)
        net_1.reset_classifier(NUM_CLASSES_DICT[args.dataset_1])
        net_1 = torch.nn.DataParallel(net_1,device_ids=[0]).cuda()
        print("M1 Uses PRETRAINED CLIP.")
        m1_name = "scratch" 
    elif "scratch-rn50" in args.model_1_ckpt: #load a image-net pretrained clip only.
        net_1 = timm.create_model(args.arch,pretrained=False)
        net_1.reset_classifier(NUM_CLASSES_DICT[args.dataset_1]) ##TODO: assumed to have same # of classes!
        net_1 = load_moco_ckpt(model=net_1, pretrained_ckpt="/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar")
        net_1 = torch.nn.DataParallel(net_1,device_ids=[0]).cuda()
        print("M1 Uses PRETRAINED MOCOV2.")
        m1_name = "scratch" 
    else:
        print("ERROR ERROR ERROR; INVALID COMBINATION SPECIFIED!")
        exit()
    """
    Create (optional) Model 2
    """
   
    if "clip" in args.model_2_ckpt and "scratch" not in args.model_2_ckpt:
        encoder_type = args.arch.split("-")[-1]
        print("\t => Clip Encoder: ",encoder_type)
        net_2 = clip_model.ClipModel(model_name=encoder_type,scratch=True)
        net_2.reset_classifier(NUM_CLASSES_DICT[args.dataset_1])
        net_2 = torch.nn.DataParallel(net_2,device_ids=[0]).cuda()
        ckpt = torch.load(args.model_2_ckpt)
        incomp, unexpected = net_2.load_state_dict(ckpt['state_dict'])
        print("LOADED M2 CKPT")
        print("Incompatible Keys: ",incomp)
        print("Unexpected Keys: ",unexpected)
        m2_name = args.model_2_ckpt
    elif "scratch-clip" in args.model_2_ckpt: #load a image-net pretrained clip only.
        encoder_type = args.arch.split("-")[-1]
        print("\t => Clip Encoder: ",encoder_type)
        net_2 = clip_model.ClipModel(model_name=encoder_type,scratch=False)
        net_2.reset_classifier(NUM_CLASSES_DICT[args.dataset_1])
        net_2 = torch.nn.DataParallel(net_2,device_ids=[0]).cuda()
        print("Model 2, PRETRAINED CLIP.")
        m2_name = args.model_2_ckpt  
    elif args.model_2_ckpt.lower() == "scratch-rn50":
        net_2 = timm.create_model(args.arch,pretrained=False)
        net_2.reset_classifier(NUM_CLASSES_DICT[args.dataset_1]) ##TODO: assumed to have same # of classes!
        net_2 = load_moco_ckpt(model=net_2, pretrained_ckpt="/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar")
        net_2 = torch.nn.DataParallel(net_2,device_ids=[0]).cuda()
        print("M2 Uses PRETRAINED MOCOV2.")
        m2_name = "scratch"
    
    elif "none" in args.model_2_ckpt:
        print("NO MODEL 2 Specified, just using model 1") 
        net_2 = net_1 #if model 2 is not specified, use model 1 twice.
        m2_name = args.model_1_ckpt 
    else:
        print("ERROR ERROR ERROR; INVALID COMBINATION SPECIFIED!")
        exit()
    """
    Create Dataloader 1
    """
    if args.dataset == "blendedCIFAR":
        test_dataset = blendedCIFARMNIST(
                train=False,
                randomized=False,
                correlation_strength=args.correlation_strength)
        d1_test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataset == 'blendedSTL': 
        test_dataset = blendedSTLMNIST(
                train=False,
                randomized=False,
                correlation_strength=args.correlation_strength)
        d1_test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataset == 'randSTL': 
        test_dataset = blendedSTLMNIST(
                train=False,
                randomized=True,
                correlation_strength=args.correlation_strength)
        d1_test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        print("ERROR ERROR ERROR!")
    """
    Create (optional) Dataloader 2
    """
    if args.dataset_2.lower() != "none":
        print("**** Creating Second Dataset! ")
        d2_train_transform = get_transform(dataset=args.dataset_2, SELECTED_AUG=args.dataset_2_trainaug,use_clip_mean=True) 
        d2_test_transform = get_transform(dataset=args.dataset_2, SELECTED_AUG=args.dataset_2_testaug,use_clip_mean=True) 
        d2_train_loader, d2_test_loader = get_dataloaders(dataset=args.dataset_2,
            args=args, 
            train_aug=args.dataset_2_trainaug,
            test_aug=args.dataset_2_testaug, 
            train_transform=d2_train_transform,
            test_transform=d2_test_transform) 
            
    else:
        print("**** ONLY 1 Dataset was provided. Doubling Loader! ")
        d2_test_loader = d1_test_loader

    """
    Get accuracies.
    """
    _, n1_test_acc =  test(net=net_1,test_loader=d1_test_loader)
    print("=> M1: ",args.model_1_ckpt)
    print("=> M1, test acc: ",n1_test_acc)
    """
    By correctly choosing which datasets to use,
    we can extract the appropriate CKA scores for
    model vs. model, ood vs id, protocol vs. protocol.
    """
    # pdb.set_trace()
    if "clip" in args.model_1_ckpt:
        layer_names = [
            'module._model.visual.layer1.0.bn3', #1 
            'module._model.visual.layer1.1.bn3',
            'module._model.visual.layer1.2.bn3',
            'module._model.visual.layer2.0.bn3',
            'module._model.visual.layer2.1.bn3',
            'module._model.visual.layer2.2.bn3',
            'module._model.visual.layer3.0.bn3',
            'module._model.visual.layer3.1.bn3',
            'module._model.visual.layer3.2.bn3',
            'module._model.visual.layer4.0.bn3',
            'module._model.visual.layer4.1.bn3',
            'module._model.visual.layer4.2.bn3',
            # 'module.fc'
            ]
    elif "resnet50" in args.model_1_ckpt: 
        layer_names = ['module.layer1.0.bn3', #1 
        'module.layer1.1.bn3', #2 
        'module.layer1.2.bn3', #3
        'module.layer2.0.bn3', #4 
        'module.layer2.1.bn3', #5
        'module.layer2.2.bn3', #6 
        'module.layer3.0.bn3', #7
        'module.layer3.1.bn3', #8
        'module.layer3.2.bn3', #9
        'module.layer4.0.bn3', #10
        'module.layer4.1.bn3', #11
        'module.layer4.2.bn3' #12
        ]
    # pdb.set_trace()
    net_1.eval()
    net_2.eval()
    cka = CKA(net_1, net_2,
            model1_name=m1_name,   
            model2_name=m2_name,   
            model1_layers=layer_names, 
            model2_layers=layer_names,
            device='cuda:0')

    use_train = False
    if args.dataset_2.lower() == "none":
        cka.compare(dataloader1=d1_test_loader) 
        test_results = cka.export()  
    else:
        cka.compare(dataloader1=d1_test_loader,dataloader2=d2_test_loader) # secondary dataloader is optional
        test_results = cka.export() 

    """"
    Save the diagonal values to log file.
    """
    safety_logs_prefix = "/usr/workspace/trivedi1/simplicity_experiments_aaai/safety_logs"
    print("=> Save Name:",args.save_name)
    test_results_diag = np.diag(test_results['CKA'].numpy())
    test_results_diag = np.round(test_results_diag,4)
    print("=> Avg CKA: {0:.4f}".format(np.mean(test_results_diag))) 
    avg_cka = np.round(np.mean(test_results_diag),4)
    test_results_diag = test_results_diag.tolist()
    test_results_diag = [np.round(i,4) for i in test_results_diag]
    print("=> CKAs: ",test_results_diag)
    with open("{}/cka_simplicity.csv".format(safety_logs_prefix),"a") as f:
        write_str = "{save_name},{dataset}-{correlation},{avg_cka:.4f},{cka}\n".format(save_name=args.save_name,dataset=args.dataset_1,correlation=args.correlation_strength,avg_cka=avg_cka, cka=test_results_diag)
        f.write(write_str)
if __name__ == '__main__':
    main()
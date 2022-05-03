"""
Given a pre-computed index, compute prediction depth.  
"""

from collections import defaultdict
from random import shuffle
from torch_cka import CKA
import os
import time
import torch
import timm
import tqdm 
import argparse
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from utils import *
import pdb 
import random
import faiss
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

class FaissKNeighbors:
    def __init__(self, k=30):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

def arg_parser():
    parser = argparse.ArgumentParser(
    description='Compute Prediction Depth',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--index',
        type=str,
        default='tmp.idx')
     
    parser.add_argument(
        '--eval_dataset',
        type=str,
        default='ev.idx')
    
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet50',
        choices=['resnet50'])
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
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
        '--ckpt',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar'
    )
    
    parser.add_argument(
        '--dataset_trainaug',
        type=str,
        default='test', #intentional for CKA!
        choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test']
    )
    
    parser.add_argument(
        '--dataset_testaug',
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
        default='tmp'
    ) 
    
    args = parser.parse_args()
    return args  

def get_oodloader(dataset):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),normalize])
    if dataset.upper() == 'STL10':
        ood_dataset = STL10(root="/p/lustre1/trivedi1/vision_data",
            split='test',
            download=True,
            transform=transform)
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

def load_dp_ckpt(model,pretrained_ckpt):
    ckpt = torch.load(pretrained_ckpt)
    new_keys = [(k, k.replace("module.","")) for k in list(ckpt['state_dict'].keys())]
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
        else:
            print("***** ERROR ERROR ERROR ******")
            print("Invalid Dataset Selected, Exiting")
            exit()
    else:
        #augmentation will be applied in training loop! 
        normalize = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
            transforms.Normalize(norm_dict[args.dataset+'_mean'], norm_dict[args.dataset + "_std"]),
            ])
        if dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=normalize)
        elif dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
                train=True,
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
    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
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
    train_acc, test_acc = -1,-1
    
    """
    Create Model 1
    """
    net = timm.create_model(args.arch,pretrained=False)
    net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
    if "moco" in args.ckpt:
        #Can't use data-parallel for feature extractor 
        net = load_moco_ckpt(model=net, pretrained_ckpt=args.ckpt).cuda()
    else:
        net = load_dp_ckpt(net,ckpt=args.ckpt).cuda() #do this so there is not a conflict. 

    """
    Load the index and create a kNN Probe.
    """
    index = faiss.read_index(args.index)

    """
    Create Dataloader
    """
    train_transform = get_transform(dataset=args.eval_dataset, SELECTED_AUG=args.dataset_trainaug) 
    test_transform = get_transform(dataset=args.eval_dataset, SELECTED_AUG=args.dataset_testaug) 
    train_loader, test_loader = get_dataloaders(dataset=args.dataset,
        args=args, 
        train_aug=args.dataset_trainaug,
        test_aug=args.dataset_testaug, 
        train_transform=train_transform,
        test_transform=test_transform) 

    """
    Get accuracies.
    """
    ood_loader = get_oodloader(dataset='stl10')
    __, ood_test_acc =  test(net=net,test_loader=ood_loader)
    _, test_acc =  test(net=net,test_loader=test_loader)
    _, train_acc =  test(net=net,test_loader=train_loader)
    print("********** Accs *************")
    print("Save Name: ",args.save_name) 
    print("Train Acc: ", train_acc)
    print("Test Acc: ", test_acc)
    print("OOD Acc.: ",ood_test_acc)
    print("*****************************") 
    
    """
    Create feature extractor 
    so that we may compute predictions.
    Note: we don't need to save the features here 
    just their predictions, if there is a space restriction.
    """
    layer_names = ['layer1.0.bn3','layer1.0.add','layer1.0.act3','layer1.1.bn3','layer1.1.add','layer1.1.act3']
    feature_extractor = create_feature_extractor(net, return_nodes=layer_names)
    features_dict = defaultdict(list) 
    labels_list = []
    with torch.no_grad():
        for data,targets in tqdm.tqdm(test_loader):
            data, targets= data.cuda(), targets.cuda()
            features = feature_extractor(data) #this should be the key of the desired layer
            [features_dict[k].append(features[k].detach().cpu().numpy()) for k in features.keys()] 
            labels_list.append(targets.detach().cpu().numpy())

        labels = np.concatenate(labels_list)

    """
    Once we have the predictions for each sample. 
    We need to create a matrix of predictions: N x Depth
    Then, we must find the instance after which the prediction is the same
    with the last layer.
    """
    for i,l in enumerate(layer_names):
        fknn = FaissKNeighbors(k=30)
        fknn.fit(X=features_dict[l],y=labels)
        faiss.write_index(fknn.index,"{sn}_{l}.idx".format(args.save_name,l))
        preds = fknn.predict(X=features_dict[l])
        acc = np.sum(preds == labels) / len(preds) 
        print("Idx {0} name:{1}_{2}: {3:.4f}".format(i,args.save_name,l,acc))

    """
    Extract features   
    """

if __name__ == '__main__':
    main()
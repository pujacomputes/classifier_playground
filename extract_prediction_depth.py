"""
Create and save the index.
The index can then be used to 
perform knn predictions on other datasets. 
"""
import math
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
        self.index.add(X)
        self.y = y

    def predict(self, X):
        _, indices = self.index.search(X, k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

def arg_parser():
    parser = argparse.ArgumentParser(
    description='Extract Faiss',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10'])
     
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
    net_name = args.ckpt 

    """
    Create Dataloader
    """
    train_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.dataset_trainaug) 
    test_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.dataset_testaug) 
    train_loader, test_loader = get_dataloaders(dataset=args.dataset,
        args=args, 
        train_aug=args.dataset_trainaug,
        test_aug=args.dataset_testaug, 
        train_transform=train_transform,
        test_transform=test_transform) 
    test_batch = test_loader.dataset[0][0].unsqueeze(0).cuda()
    """
    Get accuracies.
    """
    # ood_loader = get_oodloader(dataset='stl10')
    # __, ood_test_acc =  test(net=net,test_loader=ood_loader)
    # _, test_acc =  test(net=net,test_loader=test_loader)
    # _, train_acc =  test(net=net,test_loader=train_loader)
    # print("********** Accs *************")
    # print("Save Name: ",args.save_name) 
    # print("Train Acc: ", train_acc)
    # print("Test Acc: ", test_acc)
    # print("OOD Acc.: ",ood_test_acc)
    # print("*****************************") 
    
    
    """
    Going to create a feature extractor
    that will allow us to create different kNN probes.
    Please note, this is very memory intensive. 
    Its probably best to have a small list of layer names or find a way to the reduce the size of features.
    Alternatively, accept it and create a huge np memmap. 
    """
    net.eval()
    layer_names = 'fc' #layer1.0.add','layer1.0.act3','layer1.1.bn3','layer1.1.add','layer1.1.act3']
    feature_extractor = create_feature_extractor(net, return_nodes=[layer_names])
    f = feature_extractor(test_batch)[layer_names].detach().cpu().numpy()
    print("Representation Shape: ",f.shape,np.prod(f.shape[1:]))
    arr_shape = (len(train_loader.dataset),np.prod(f.shape[1:]))

    print("Array Shape: ",arr_shape)
    labels_list = []
    #We are going to preallocate a very large array....
    feat_array = np.memmap('tmp.npy',dtype='float32', mode='w+', shape=arr_shape) 
    idx_counter = 0
    with torch.no_grad():
        for data,targets in tqdm.tqdm(train_loader):
            bs = len(targets)
            data, targets= data.cuda(), targets.cuda()
            features = feature_extractor(data) 
            ### extract the desired layers from the features dict. We are passing in the flattened version!
            feat_array[idx_counter:idx_counter+bs] = features[layer_names].reshape(bs,-1).detach().cpu().numpy()
            labels_list.append(targets.detach().cpu().numpy())
            idx_counter += bs
        labels = np.concatenate(labels_list)
    
    """
    Having acquired the arrays, we can now create and save the probes. 
    Note! Please make sure there is no shuffling prior to use.
    """
    print("Creating KNN Classifier....")
    fknn = FaissKNeighbors(k=30)
    fknn.fit(X=feat_array,y=labels)
    print("\t Done Fitting.")
    faiss.write_index(fknn.index,"{0}_{1}.idx".format(args.save_name,layer_names))
    preds = fknn.predict(X=feat_array)
    acc = np.sum(preds == labels) / len(preds) 
    print("{0}, {1}: {2:.4f}".format(args.save_name,layer_names,acc))

    """
    Once the probes are created and saved 
    for all relevant layers. We need to get the
    kNN prediction at each layer and confirm when
    the prediction stops changing.
    """

if __name__ == '__main__':
    main()
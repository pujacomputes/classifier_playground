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
    def __init__(self, k=30,d=100):
        self.y = []
        self.k = k
        self.d = d
        
    def make_index(self,d):
        self.index = faiss.IndexFlatL2(d) 

    def add(self, X):
        self.index.add(X)

    def predict(self, X):
        _, indices = self.index.search(X, k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

class FaissKNeighborsLSH:
    def __init__(self, k=30,d=100):
        self.y = []
        self.k = k
        self.d = d
        
    def make_index(self,d):
        self.index = faiss.IndexLSH(d,8096) 

    def add(self, X):
        self.index.add(X)

    def predict(self, X):
        _, indices = self.index.search(X, k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
def check_last(z):
    return z == z[-1]

#https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions

            i_x = ia[i]
            pred_depth = p[np.where(i_x == True)[0].max()]
            return pred_depth

def get_pred_depth(loader, probe_dict, feature_extractor, features_short_list, no_avg_list,eval_batch_size=128):
    correct = np.zeros(len(features_short_list))
    pred_depth = np.zeros((eval_batch_size, len(features_short_list)))
    
    dataset_model_preds = [] 
    dataset_pred_depths = [] 
    with torch.no_grad():
        for data,targets in tqdm.tqdm(loader):
            bs = len(targets)
            data, targets = data.cuda(), targets.numpy()
            features = feature_extractor(data)
            if bs != eval_batch_size or bs != pred_depth.shape[0]:
                pred_depth = np.zeros((bs, len(features_short_list))) 
            else:
                pred_depth.fill(-1)
            for enum,l in enumerate(features_short_list):
                if l in no_avg_list: 
                    f= features[l].reshape(bs,-1).detach().cpu().numpy()
                else:
                    f= torch.nn.functional.avg_pool2d(features[l],kernel_size=features[l].shape[-1]).reshape(bs,-1).detach().cpu().numpy()
                """
                Record the predictions at each depth for each sample.
                """
                preds = probe_dict[l].predict(f)
                pred_depth[:,enum] = preds 
                correct[enum] += np.sum(preds == targets) 
            """
            Having recorded the pred at each depth.
            Determine the prediction depth.
            """
            bool_pred_depth = np.apply_along_axis(check_last,1,pred_depth) 
            dataset_pred_depths.extend(np.apply_along_axis(rle,1,bool_pred_depth)) 
    return correct, dataset_pred_depths

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
        default='/p/lustre1/trivedi1/compnets/classifier_playground/ckpts/ft+cifar10_resnet50_ft_test_test_0_-1.0_-1.0_20_0.0003_0.0_-1.0_0_ft_model_best.pth.tar'
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
        ood_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
            split='test',
            download=False,
            transform=transform)

        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        ood_dataset.labels = stl_to_cifar_indices[ood_dataset.labels]
        ood_dataset = torch.utils.data.Subset(ood_dataset,np.where(ood_dataset.labels != -1)[0])
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
        batch_size=128,
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
        elif dataset == 'STL10' or dataset == 'stl10':
            train_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
                split='train',
                download=False,
                transform=train_transform)

            stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
            train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
            train_dataset = torch.utils.data.Subset(train_dataset,np.where(train_dataset.labels != -1)[0])
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
        elif dataset == 'STL10' or dataset == 'stl10':
            train_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
                split='train',
                download=False,
                transform=normalize)

            stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
            train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
            train_dataset = torch.utils.data.Subset(train_dataset,np.where(train_dataset.labels != -1)[0])
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
    train_acc, test_acc,ood_test_acc = -1,-1,-1

    print("=> Save Name: ",args.save_name) 
    """
    Create Model 1
    """
    net = timm.create_model(args.arch,pretrained=False)
    net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
    if "moco" in args.ckpt:
        #Can't use data-parallel for feature extractor 
        net = load_moco_ckpt(model=net, pretrained_ckpt=args.ckpt).cuda()
    else:
        net = load_dp_ckpt(net,pretrained_ckpt=args.ckpt).cuda() #do this so there is not a conflict. 
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
    
    net.eval()
    features_short_list = ['x','layer1.0.bn1','layer1.1.add','layer1.2.act3',
        'layer2.0.bn1','layer2.2.add','layer2.3.act3','layer3.0.bn1',
        'layer3.4.add','layer3.5.act3','layer4.0.bn1','layer4.1.add',
        'layer4.2.act3','global_pool.flatten','fc']   
    no_avg_list = ['global_pool.flatten','fc']   
    targets = train_loader.dataset.targets
    feature_extractor = create_feature_extractor(net, return_nodes=features_short_list)
    probe_dict = {} 
    test_feat_dict = feature_extractor(test_batch)
    for l in features_short_list: 
        if l in no_avg_list: 
            features = test_feat_dict[l].reshape(1,-1).detach().cpu().numpy()
        else:
            features = torch.nn.functional.avg_pool2d(test_feat_dict[l],kernel_size=test_feat_dict[l].shape[-1]).reshape(1,-1)
        d = int(features.shape[1])
        probe = FaissKNeighbors(k=30,d=d)
        probe.make_index(d=d)
        probe.y = np.asarray(train_loader.dataset.targets)
        probe_dict[l] = probe
    """
    Create each of the probes.
    """
    with torch.no_grad():
        for data,targets in tqdm.tqdm(train_loader):
            bs = len(targets)
            data, _ = data.cuda(), targets.cuda()
            features = feature_extractor(data)
            for l in features_short_list:
                if l in no_avg_list: 
                    f = features[l].reshape(bs,-1).detach().cpu().numpy()
                else:
                    f = torch.nn.functional.avg_pool2d(features[l],kernel_size=features[l].shape[-1]).reshape(bs,-1).detach().cpu().numpy()
                probe_dict[l].add(X=f)
        print("Finished Creating KNN Classifiers....")
        
    """
    Get the prediction depths
    """
    ckpt = {
        'train_acc':train_acc,
        'test_acc':test_acc,
        'ood_test_acc':ood_test_acc,
        'id_dataset':'cifar10',
        'ood_dataset':'stl10',
        'id_dataset':args.dataset,
    }
    for name, loader in zip(['id_train','id_test','ood_test'],[train_loader,test_loader,ood_loader]):
        correct_num, pred_list = get_pred_depth(loader=loader,
            probe_dict=probe_dict, 
            feature_extractor = feature_extractor, 
            features_short_list = features_short_list, 
            no_avg_list = no_avg_list,
            eval_batch_size=128)
        print("************ {} Summary **************".format(name))
        p_v, p_c= np.unique(pred_list, return_counts=True)
        for v,cx in zip(p_v, p_c):
            per_layer = np.round(100 * (cx/len(pred_list)),4)
            print("\t{},{}".format(int(v),per_layer))
        print("**********************************************")
        ckpt["{}_pred_depth".format(name)] = pred_list
        ckpt["{}_correct_num".format(name)] = correct_num/len(loader.dataset) 
        print("{0} Acc: {1}".format(name,correct_num/len(loader.dataset)))
    """
    Save the idx and prediction depths.
    """
    save_path = "/p/lustre1/trivedi1/compnets/classifier_playground/knn_probes/{}".format(args.save_name)
    print("=> Save Path: ",save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    s_k = "{}/pred_depth.ckpt".format(save_path)
    torch.save(obj=ckpt,f=s_k)
if __name__ == '__main__':
    main()
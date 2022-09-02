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
from utils import get_transform, NUM_CLASSES_DICT, test,norm_dict 
import pdb 
import random
from clip_model import ClipModel
from blended_cifar_mnist import blendedCIFARMNIST, blendedSTLMNIST
def arg_parser():
    parser = argparse.ArgumentParser(
    description='Extract CKA',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_1',
        type=str,
        default='cifar10',
        choices=['cifar10','stl10','STL10','cifar100','blendedcifar10','blendedcifar','blendedstl'])
    
    parser.add_argument(
        '--dataset_2',
        type=str,
        default='stl10',
        choices=['stl10','none'])
    
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
        '--model_1_ckpt',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar'
    )
    
    parser.add_argument(
        '--model_2_ckpt',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar'
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
    
    args = parser.parse_args()
    return args  

def get_oodloader(args,dataset):
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
    elif dataset.lower() == 'blendedstl10':
        ood_dataset = blendedSTLMNIST(train=False, transform=None)
        ood_loader = torch.utils.data.DataLoader(
                ood_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True) 
    elif dataset.lower() == 'blendedrandstl10':
        random_ood_dataset = blendedSTLMNIST(train=False, transform=None,randomized=True)
        ood_loader = torch.utils.data.DataLoader(
                random_ood_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True) 
    
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
    print("DATASET: ",dataset)
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
        elif dataset.lower() == 'blendedstl10':
            train_dataset = blendedSTLMNIST(train=True, transform=None)
        elif dataset.lower() == 'blendedrandstl10':
            train_dataset = blendedSTLMNIST(train=True, transform=None,randomized=True)
        elif dataset.lower() == 'blendedcifar10':
            train_dataset = blendedCIFARMNIST(train=True, transform=None,correlation_strength=1.0)
        elif dataset.lower() == 'blendedrandcifar10':
            train_dataset = blendedCIFARMNIST(train=True, transform=None,randomized=True)
        else:
            print("***** ERROR ERROR ERROR ******")
            print("=> Invalid Dataset Selected, Exiting")
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
        elif dataset.lower() == 'blendedstl10':
            train_dataset = blendedSTLMNIST(train=True, transform=None)
        elif dataset.lower() == 'blendedrandstl10':
            train_dataset = blendedSTLMNIST(train=True, transform=None,randomized=True)
        elif dataset.lower() == 'blendedcifar10':
            train_dataset = blendedCIFARMNIST(train=True, transform=None,correlation_strength=1.0)
        elif dataset.lower() == 'blendedrandcifar10':
            train_dataset = blendedCIFARMNIST(train=True, transform=None,randomized=True)

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
    
    elif dataset.lower() == 'blendedstl10':
        test_dataset = blendedSTLMNIST(train=False, transform=None)
    elif dataset.lower() == 'blendedrandstl10':
        test_dataset = blendedSTLMNIST(train=False, transform=None,randomized=True)
    elif dataset.lower() == 'blendedcifar10':
        test_dataset = blendedCIFARMNIST(train=False, transform=None,correlation_strength=1.0)
    elif dataset.lower() == 'blendedrandcifar10':
        test_dataset = blendedCIFARMNIST(train=False, transform=None,randomized=True)

    
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
    n1_d1_test_acc,n1_d1_train_acc,n1_d2_train_acc, n1_d2_test_acc = -1,-1,-1,-1
    n2_d1_test_acc,n2_d1_train_acc,n2_d2_train_acc, n2_d2_test_acc = -1,-1,-1,-1
    print("args.dataset_1",args.dataset_1) 
    
    use_clip_mean = "clip" in args.arch 
    """
    Create Model 1
    """
    net_1 = timm.create_model(args.arch,pretrained=False)
    net_1.reset_classifier(NUM_CLASSES_DICT[args.dataset_1])
    if "moco" in args.model_1_ckpt:
        net_1 = load_moco_ckpt(model=net_1, pretrained_ckpt=args.model_1_ckpt)
        net_1 = torch.nn.DataParallel(net_1,device_ids=[0]).cuda()
    else:
        net_1 = torch.nn.DataParallel(net_1,device_ids=[0]).cuda()
        ckpt = torch.load(args.model_1_ckpt)
        net_1.load_state_dict(ckpt['state_dict'])
    m1_name = args.model_1_ckpt 
    """
    Create (optional) Model 2
    """
    if args.model_2_ckpt.lower() == "scratch":
        net_2 = timm.create_model(args.arch,pretrained=False)
        net_2.reset_classifier(NUM_CLASSES_DICT[args.dataset_1]) ##TODO: assumed to have same # of classes!
        net_2 = load_moco_ckpt(model=net_2, pretrained_ckpt="/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar")
        net_2 = torch.nn.DataParallel(net_2,device_ids=[0]).cuda()
        print("M2 Uses PRETRAINED MOCOV2.")
        m2_name = "scratch" 
    elif "none" in args.model_2_ckpt.lower() :
        print("NO MODEL 2 Specified, just using model 1") 
        net_2 = net_1 #if model 2 is not specified, use model 1 twice.
        m2_name = args.model_1_ckpt 
    else:
        net_2 = timm.create_model(args.arch,pretrained=False)
        net_2.reset_classifier(NUM_CLASSES_DICT[args.dataset_1]) ##TODO: assumed to have same # of classes!
        net_2 = torch.nn.DataParallel(net_2,device_ids=[0]).cuda()
        ckpt = torch.load(args.model_2_ckpt)
        net_2.load_state_dict(ckpt['state_dict'])
        m2_name = args.model_2_ckpt 
   

    use_clip_mean_m2 = "clip" in args.model_2_ckpt
    use_clip_mean_m1 = "clip" in args.model_1_ckpt

    """
    Create Dataloader 1
    """
    d1_train_transform = get_transform(dataset=args.dataset_1, SELECTED_AUG=args.dataset_1_trainaug,use_clip_mean=use_clip_mean_m1) 
    d1_test_transform = get_transform(dataset=args.dataset_1, SELECTED_AUG=args.dataset_1_testaug,use_clip_mean=use_clip_mean_m1) 
    d1_train_loader, d1_test_loader = get_dataloaders(dataset=args.dataset_1,
        args=args, 
        train_aug=args.dataset_1_trainaug,
        test_aug=args.dataset_1_testaug, 
        train_transform=d1_train_transform,
        test_transform=d1_test_transform) 
    """
    Create (optional) Dataloader 2
    """
    if args.dataset_2.lower() != "none":
        print("=> Loading Second Dataset!")
        d2_train_transform = get_transform(dataset=args.dataset_2, SELECTED_AUG=args.dataset_2_trainaug,use_clip_mean=use_clip_mean_m2) 
        d2_test_transform = get_transform(dataset=args.dataset_2, SELECTED_AUG=args.dataset_2_testaug,use_clip_mean=use_clip_mean_m2) 
        d2_train_loader, d2_test_loader = get_dataloaders(dataset=args.dataset_2,
            args=args, 
            train_aug=args.dataset_2_trainaug,
            test_aug=args.dataset_2_testaug, 
            train_transform=d2_train_transform,
            test_transform=d2_test_transform) 
            
    else:
        d2_train_loader = d1_train_loader
        d2_test_loader = d1_test_loader

    """
    Get accuracies.
    """
    if "blended" in args.dataset_1:
        print("=> Using Blended OOD")
        ood_loader = get_oodloader(args,dataset='blendedstl10')
    else:
        ood_loader = get_oodloader(args,dataset='stl10')

    n1_ood_test_acc, n2_ood_test_acc = -1, -1
    if args.dataset_1 != 'cifar100':
        _, n1_ood_test_acc =  test(net=net_1,test_loader=ood_loader)
        _, n2_ood_test_acc =  test(net=net_2,test_loader=ood_loader)
    _, n1_d1_test_acc =  test(net=net_1,test_loader=d1_test_loader)
    _, n2_d1_test_acc =  test(net=net_2,test_loader=d1_test_loader)
    
    print("=> M1: ",args.model_1_ckpt)
    print("M1,OOD Acc.: ",n1_ood_test_acc)
    print("M1,D1 Acc.: ",n1_d1_test_acc)
    
    print("=> M2: ",args.model_2_ckpt)
    print("M2,OOD Acc.: ",n2_ood_test_acc)
    print("M2,D1 Acc.: ",n2_d1_test_acc)
    
    """
    By correctly choosing which datasets to use,
    we can extract the appropriate CKA scores for
    model vs. model, ood vs id, protocol vs. protocol.
    """
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
        if use_train:
            cka.compare(dataloader1=d1_train_loader) 
            train_results = cka.export() 
    else:
        print()
        print("*"*30)
        print("Comparing different loaders!") 
        print("*"*30)

        cka = CKA(net_1, net_2,
                model1_name=m1_name,   
                model2_name="tmp",   
                model1_layers=layer_names, 
                model2_layers=layer_names,
                device='cuda:0')
        cka.compare(dataloader1=d1_test_loader,dataloader2=d2_test_loader) # secondary dataloader is optional
        test_results = cka.export() 
        if use_train: 
            cka.compare(dataloader1=d1_train_loader,dataloader2=d2_train_loader) 
            train_results = cka.export() 
    print("********** Test ************") 
    print(test_results)
    print("*****************************")
    print()
    if use_train:
        print("********** Train ************") 
        print(train_results)
        print("*****************************") 
   
    """"
    Save the diagonal values to log file.
    """
    if args.dataset_1 == 'cifar100':
        safety_logs_prefix = "/usr/workspace/trivedi1/cifar100_experiments_aaai/resnet50_safety_logs"
    elif args.dataset_1 == 'cifar10': 
        safety_logs_prefix = "/p/lustre1/trivedi1/compnets/classifier_playground/safety_logs"
    print("=> Save Name:",args.save_name)
    test_results_diag = np.diag(test_results['CKA'].numpy())
    test_results_diag = np.round(test_results_diag,4)
    print("=> Avg CKA: {0:.4f}".format(np.mean(test_results_diag))) 
    avg_cka = np.round(np.mean(test_results_diag),4)
    test_results_diag = test_results_diag.tolist()
    test_results_diag = [np.round(i,4) for i in test_results_diag]
    print("=> CKAs: ",test_results_diag)
    with open("{}/cka.csv".format(safety_logs_prefix),"a") as f:
        write_str = "{save_name},{avg_cka:.4f},{cka}\n".format(save_name=args.save_name,avg_cka=avg_cka, cka=test_results_diag)
        f.write(write_str)
if __name__ == '__main__':
    main()
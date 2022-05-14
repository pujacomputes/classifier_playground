import argparse
import random
import numpy as np
import tqdm
import timm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from cifar10p1 import CIFAR10p1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES_DICT = {
    'cifar10':10,
    'stl10':10,
    'none':-1,
    'STL10':10,

}

norm_dict = {
    'cifar10_mean':[0.485, 0.456, 0.406],
    'cifar10_std': [0.228, 0.224, 0.225],
    'stl10_mean':[0.485, 0.456, 0.406],
    'stl10_std': [0.228, 0.224, 0.225],
}

#https://github.com/AnanyaKumar/transfer_learning/blob/main/unlabeled_extrapolation/baseline_train.py
def get_param_weights_counts(net, detach):
    weight_dict = {}
    count_dict = {}
    for param in net.named_parameters():
        name = param[0]
        weights = param[1]
        if detach:
            weight_dict[name] = weights.detach().clone()
        else:
            weight_dict[name] = weights
        count_dict[name] = np.prod(np.array(list(param[1].shape)))
    return weight_dict, count_dict

def get_l2_dist(weight_dict1, weight_dict2, ignore='.fc.'):
    l2_dist = torch.tensor(0.0).cuda()
    for key in weight_dict1:
        if ignore not in key:
            l2_dist += torch.sum(torch.square(weight_dict1[key] - weight_dict2[key]))
    return l2_dist

def get_oodloader(args,dataset):
    normalize = transforms.Normalize(norm_dict[args.dataset+"_mean"], norm_dict[args.dataset + "_std"])
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),normalize])
    if dataset.upper() == 'STL10':
        ood_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
            split='test',
            download=False,
            transform=transform)

        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        ood_dataset.labels = stl_to_cifar_indices[ood_dataset.labels]
        ood_dataset = torch.utils.data.Subset(ood_dataset,np.where(ood_dataset.labels != -1)[0])
    
    elif dataset.upper() == "CIFAR10.1":
        ood_dataset = CIFAR10p1(root="/p/lustre1/trivedi1/vision_data/CIFAR10.1/",
            split='test',
            verision='v6',
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
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=wif) 
    return ood_loader

def get_transform(dataset,SELECTED_AUG):
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'imagenet1K':
        num_classes = 1000
    elif dataset.lower() == 'stl10':
        num_classes = 10
    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
        exit()
    hparams= {'translate_const': 100, 'img_mean': (124, 116, 104)}
    normalize = transforms.Normalize(norm_dict[dataset+"_mean"], norm_dict[dataset + "_std"])
    resize = transforms.Resize(size=224)
    if SELECTED_AUG == 'cutout':
        transform = timm.data.random_erasing.RandomErasing(probability=1.0, 
            min_area=0.02, 
            max_area=1/3, 
            min_aspect=0.3, 
            max_aspect=None,
            mode='const', 
            min_count=1, 
            max_count=None, 
            num_splits=0, 
            device=DEVICE)
    elif SELECTED_AUG == 'mixup':
        #mixup active if mixup_alpha > 0
        #cutmix active if cutmix_alpha > 0
        transform = timm.data.mixup.Mixup(mixup_alpha=1., 
            cutmix_alpha=0., 
            cutmix_minmax=None, 
            prob=1.0, 
            switch_prob=0.0,
            mode='batch', 
            correct_lam=True, 
            label_smoothing=0.1, 
            num_classes=num_classes)
    elif SELECTED_AUG == 'cutmix':
        transform = timm.data.mixup.Mixup(mixup_alpha=0., 
            cutmix_alpha=1., 
            cutmix_minmax=None, 
            prob=0.8, 
            switch_prob=0.0,
            mode='batch', 
            correct_lam=True, 
            label_smoothing=0.1, 
            num_classes=num_classes)
    elif SELECTED_AUG == 'autoaug':
        #no searching code;
        transform = timm.data.auto_augment.auto_augment_transform(config_str='original-mstd0.5',
        hparams= hparams)
    elif SELECTED_AUG == 'augmix':
        transform = timm.data.auto_augment.augment_and_mix_transform(
            config_str='augmix-m5-w4-d2',
            hparams=hparams)
    elif SELECTED_AUG == 'randaug':
        transform = timm.data.auto_augment.rand_augment_transform( 
            config_str='rand-m3-n2-mstd0.5',
            hparams=hparams
            )
    elif SELECTED_AUG == 'base':
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),resize, transforms.ToTensor(),normalize])
    
    elif SELECTED_AUG == 'test' or SELECTED_AUG == 'vat':
        transform = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(),normalize])
    elif SELECTED_AUG == 'pixmix':
        print("Not Implemented yet!")

    if SELECTED_AUG in ['randaug','augmix','autoaug']:
       transform = transforms.Compose([resize, transform,transforms.ToTensor(),normalize]) 
    return transform

def get_dataloaders(args,train_aug, test_aug, train_transform,test_transform,use_ft=False):
    if train_aug not in ['mixup','cutmix','cutout']:
        #initialize the augmentation directly in the dataset
        if args.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=train_transform)
        elif args.dataset == 'cifar100':
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
        if args.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=normalize)
        elif args.dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=normalize)

    """
    Create Test Dataloaders.
    """
    if args.dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=test_transform) 
        NUM_CLASSES=10
    elif args.dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=test_transform) 
        NUM_CLASSES=100
    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
        exit()
         
    # Fix dataloader worker issue
    # https://github.com/pytorch/pytorch/issues/5059
    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    if use_ft:
        batch_size = args.ft_batch_size
    else:
        batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=wif)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    return train_loader, test_loader

def train(net, train_loader, optimizer, scheduler,transform=None,transform_name=None,protocol='lp',l2sp=False):
    """Train for one epoch."""
    if protocol in ['lp','lp+ft']:
        #model needs to be eval mode!
        net.eval()
    else:
        net.train()
    loss_ema = 0.
    if transform_name in ['cutmix','mixup']:
        criterion = torch.nn.SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    for _, (images, targets) in enumerate(train_loader):

        optimizer.zero_grad()
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        #use cutmix or mixup
        if transform:
            if transform_name in ['cutmix','mixup']:
                images, targets= transform(images,target=targets)
            if transform_name == 'cutout':
                images = transform(images)
        logits = net(images)
        loss = criterion(logits, targets) 
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    return loss_ema


def test(net, test_loader, adv=None):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            # adversarial
            if adv:
                images = adv(net, images, targets)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
    return np.round(total_loss / len(test_loader),5), np.round(total_correct / len(test_loader.dataset),5)

def freeze_layers_for_lp(model):
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    return model

def unfreeze_layers_for_ft(model):
    for param in model.parameters():
        param.requires_grad = True 
    return model

def unfreeze_layers_for_lpfrz_ft(model):
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = True 
    return model

def arg_parser():
    parser = argparse.ArgumentParser(
    description='Transfer Learning Experiments',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10'])
    
    parser.add_argument(
        '--eval_dataset',
        type=str,
        default='stl10',
        choices=['stl10','cifar10.1'])
    
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet50',
        choices=['resnet50'])
    
    parser.add_argument(
        '--protocol',
        type=str,
        default='lp',
        choices=['lp', 'ft', 'lp+ft','lpfrz+ft','sklp','sklp+ft','vatlp','vatlp+ft'])
    parser.add_argument(
        '--pretrained_ckpt',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar'
    )
    """
    Linear Probe Args
    """
    parser.add_argument(
        '--train_aug',
        type=str,
        default='autoaug',
        choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test','vat']
    )
    parser.add_argument(
        '--test_aug',
        type=str,
        default='test'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=1
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=1e-1
    )
    
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
        '--learning-rate',
        '-lr',
        type=float,
        default=0.1,
        help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument(
        '--decay',
        '-wd',
        type=float,
        default=0.0005,
        help='Weight decay (L2 penalty).')
    parser.add_argument(
        '--droprate', default=0.0, type=float, help='Dropout probability')
    parser.add_argument(
        '--save',
        '-s',
        type=str,
        default='./ckpts',
        help='Folder to save checkpoints.')
    parser.add_argument(
        '--resume',
        '-r',
        type=str,
        default='',
        help='Checkpoint path for resume / test.')
    parser.add_argument('--evaluate', action='store_true', help='Eval only.')
    parser.add_argument(
        '--print-freq',
        type=int,
        default=50,
        help='Training loss print frequency (batches).')
    """
    Finetuning args
    """
    parser.add_argument(
        '--ft_train_aug',
        type=str,
        default='autoaug',
        choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test']
    )
    parser.add_argument(
        '--ft_test_aug',
        type=str,
        default='test'
    )
    parser.add_argument(
        '--ft_epochs',
        type=int,
        default=20
    )
    
    parser.add_argument(
        '--ft_batch_size',
        type=int,
        default=64
    )

    parser.add_argument(
        '--ft_learning-rate',
        '-ft_lr',
        type=float,
        default=0.1,
        help='Initial learning rate.')
    parser.add_argument(
        '--l2sp_weight',
        '-l2',
        type=float,
        default=-1,
        help='Keep LP close to self.')
    
    parser.add_argument(
        '--resume_lp_ckpt',
        type=str,
        default="None",
        help='Optional ckpt to begin LP + FT from.')
    parser.add_argument('--ft_momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument(
        '--ft_decay',
        '-ft_wd',
        type=float,
        default=0.0005,
        help='Weight decay (L2 penalty).')
    args = parser.parse_args()
    return args 

def load_moco_ckpt(model,args):
    ckpt = torch.load(args.pretrained_ckpt)
    new_keys = [(k, k.replace("module.encoder_q.","")) for k in list(ckpt['state_dict'].keys())]
    for old_k, new_k in new_keys:
        ckpt['state_dict'][new_k] = ckpt['state_dict'].pop(old_k)
    incompatible, unexpected = model.load_state_dict(ckpt['state_dict'],strict=False)
    print("Incompatible Keys: ", incompatible)
    print("Unexpected Keys: ",unexpected)
    return model

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#https://github.com/PatrickHua/SimSiam/blob/main/optimizers/lr_scheduler.py
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

class SoftTargetCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
"""
Used for getting the dataloaders 
for extracting CKA, Prediction Depth 
and evaluation. 
There is no shuffling and other parameters are 
fixed too. 
"""

def get_fixed_dataloaders(args,dataset, train_aug, train_transform):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    if train_aug != 'test':
        print("*"*65)
        print("Calling Evaluation Dataloaders with an Aug: {}!".format(train_aug))
        print("Please make sure this what you really wanted to do!")
        print("*"*65)
    normalize = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])])    
    if train_aug not in ['mixup','cutmix','cutout']:
        train_transform_x = train_transform
    else:
        train_transform_x = normalize 
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
            train=True,
            transform=train_transform_x)
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
            train=True,
            transform=train_transform_x)
    elif dataset == 'STL10' or dataset == 'stl10':
        train_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
            split='train',
            download=False,
            transform=train_transform_x)

        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
        train_dataset = torch.utils.data.Subset(train_dataset,np.where(train_dataset.labels != -1)[0])
    
    elif dataset.lower() == 'cifar10p1':
        train_dataset = CIFAR10p1(root="/p/lustre1/trivedi1/vision_data/CIFAR10.1/",
            split='train',
            verision='v6',
            transform=train_transform_x)
    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
        exit()
    
    """
    Create Test Dataloaders.
    """
    if dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=normalize)
        NUM_CLASSES=10
    elif dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=normalize)
        NUM_CLASSES=100
    elif dataset == 'STL10' or dataset == 'stl10':
        test_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
            split='test',
            download=False,
            transform=normalize)
        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        test_dataset.labels = stl_to_cifar_indices[test_dataset.labels]
        test_dataset = torch.utils.data.Subset(test_dataset,np.where(test_dataset.labels != -1)[0])
    
    elif dataset.lower() == 'cifar10p1':
        train_dataset = CIFAR10p1(root="/p/lustre1/trivedi1/vision_data/CIFAR10.1/",
            split='test',
            verision='v6',
            transform=normalize)
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

def set_linear_layer(layer, coef, intercept):
    coef_tensor = torch.tensor(coef, dtype=layer.weight.dtype).cuda()
    bias_tensor = torch.tensor(intercept, dtype=layer.bias.dtype).cuda()
    coef_param = torch.nn.parameter.Parameter(coef_tensor)
    bias_param = torch.nn.parameter.Parameter(bias_tensor)
    layer.weight = coef_param
    layer.bias = bias_param
    return layer

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout", 
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur", 
    "lines", "sparkles", "transverse_chromatic_abberation"]
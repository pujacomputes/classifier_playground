from PIL import Image
import argparse
import random
import numpy as np
import tqdm
from safety_utils import CBAR_CORRUPTIONS_IMAGENET
import timm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from cifar10p1 import CIFAR10p1
import domainnet
from breeds import Breeds, Breeds_C
from wilds_dataset import WILDS_Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES_DICT = {
    "cifar10": 10,
    "cifar100": 100,
    "living17": 17,
    "entity30": 30,
    "stl10": 10,
    "none": -1,
    "STL10": 10,
    "domainnet-sketch": 40,
    "domainnet-art": 40,
    "domainnet-real": 40,
    "domainnet-painting": 40,
    "pairedCIFAR": 10,
    "blendedCIFAR": 10,
    "blendedcifar10": 10,
    "blendedstl10": 10,
}

norm_dict = {
    "cifar10_mean": [0.485, 0.456, 0.406],
    "cifar10_std": [0.228, 0.224, 0.225],
    "mnist_mean": [0.485, 0.456, 0.406],
    "mnist_std": [0.228, 0.224, 0.225],
    "living17_mean": [0.485, 0.456, 0.406],
    "living17_std": [0.228, 0.224, 0.225],
    "entity30_mean": [0.485, 0.456, 0.406],
    "entity30_std": [0.228, 0.224, 0.225],
    "pairedcifar10_mean": [0.485, 0.456, 0.406],
    "pairedcifar10_std": [0.228, 0.224, 0.225],
    "blendedcifar10_mean": [0.485, 0.456, 0.406],
    "blendedcifar10_std": [0.228, 0.224, 0.225],
    "cifar100_mean": [0.485, 0.456, 0.406],
    "cifar100_std": [0.228, 0.224, 0.225],
    "stl10_mean": [0.485, 0.456, 0.406],
    "stl10_std": [0.228, 0.224, 0.225],
    "clip_mean": [0.48145466, 0.4578275, 0.40821073],
    "clip_std": [0.26862954, 0.26130258, 0.27577711],
    "domainnet_mean": [0.485, 0.456, 0.406],  # from domainnet py
    "domainnet_std": [0.228, 0.224, 0.225],
    "vits8-dino_mean": [0.485, 0.456, 0.406],
    "vits8-dino_std": [0.229, 0.224, 0.225]
}

waterbirds_selector = {
    "waterbirds": None,
    "landbg-landbird": [0, 0, 0],
    "landbg-waterbird": [0, 1, 0],
    "waterbg-landbird": [1, 0, 0],
    "waterbg-waterbird": [1, 1, 0],
}


# https://github.com/AnanyaKumar/transfer_learning/blob/main/unlabeled_extrapolation/baseline_train.py
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


def get_l2_dist(weight_dict1, weight_dict2, ignore=".fc."):
    l2_dist = torch.tensor(0.0).cuda()
    for key in weight_dict1:
        if ignore not in key:
            l2_dist += torch.sum(torch.square(weight_dict1[key] - weight_dict2[key]))
    return l2_dist


def get_oodloader(args, dataset, use_clip_mean=False):
    if use_clip_mean:
        normalize = transforms.Normalize(norm_dict["clip_mean"], norm_dict["clip_std"])
    elif "vits8-dino" in args.arch:
        normalize = transforms.Normalize(norm_dict["vits8-dino_mean"], norm_dict["vits8-dino_std"])
    elif "r101-1x-sk0" in args.arch:
        normalize = torch.nn.Identity() 
    else:
        normalize = transforms.Normalize(
            norm_dict[args.dataset + "_mean"], norm_dict[args.dataset + "_std"]
        )
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    if dataset.upper() == "STL10":
        ood_dataset = torchvision.datasets.STL10(
            root="/p/lustre1/trivedi1/vision_data",
            split="test",
            download=False,
            transform=transform,
        )

        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        ood_dataset.labels = stl_to_cifar_indices[ood_dataset.labels]
        ood_dataset = torch.utils.data.Subset(
            ood_dataset, np.where(ood_dataset.labels != -1)[0]
        )

    elif dataset.upper() == "CIFAR10.1":
        ood_dataset = CIFAR10p1(
            root="/p/lustre1/trivedi1/vision_data/CIFAR10.1/",
            split="test",
            version="v6",
            transform=transform,
        )
    elif "domainnet" in dataset.lower():
        domain_name = dataset.split("-")[-1]
        ood_dataset = domainnet.DomainNet(
            domain=domain_name,
            split="test",
            root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet/",
            transform=transform,
            verbose=False,
        )
    elif dataset == "cifar100":
        """
        There is not an OOD dataset for cifar100,
        so we are just going to return None.
        """
        return None
    elif dataset == "living17":
        ood_dataset = Breeds(
            root="/usr/workspace/trivedi1/vision_data/ImageNet",
            breeds_name="living17",
            info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
            source=False,
            target=True,
            split="val",
            transform=transform,
        )
    elif dataset == "entity30":
        ood_dataset = Breeds(
            root="/usr/workspace/trivedi1/vision_data/ImageNet",
            breeds_name="entity30",
            info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
            source=False,
            target=True,
            split="val",
            transform=transform,
        )
    elif "bird" in args.dataset.lower():
        ood_dataset = WILDS_Dataset(
            dataset_name="waterbirds",
            split="test",
            root="/usr/workspace/trivedi1/vision_data/Waterbirds",
            meta_selector=waterbirds_selector[args.dataset.lower()],
            transform=transform,
            download=False,
            return_meta=False,
        )

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
        worker_init_fn=wif,
    )
    return ood_loader


def get_corrupted_loader(args, dataset, corruption_name, severity, use_clip_mean=False):
    if use_clip_mean:
        normalize = transforms.Normalize(norm_dict["clip_mean"], norm_dict["clip_std"])
    elif "vits8-dino" in args.arch:
        normalize = transforms.Normalize(norm_dict["vits8-dino_mean"], norm_dict["vits8-dino_std"])
    elif "r101-1x-sk0" in args.arch:
        normalize = torch.nn.Identity() 
    else:
        normalize = transforms.Normalize(
            norm_dict[args.dataset + "_mean"], norm_dict[args.dataset + "_std"]
        )
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    if "cifar" in dataset:
        print("**** EXITING ****")
        print("Use npy files for CIFAR-10-C")
        exit()
    elif "domainnet" in dataset:
        domain_name = dataset.split("-")[-1]
        ood_dataset = domainnet.DomainNet(
            domain=domain_name,
            split="test",
            root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet-Corrupted/{}/{}".format(
                corruption_name, severity
            ),
            transform=transform,
            verbose=False,
        )
    elif "living17" in dataset:
        if corruption_name in CORRUPTIONS:
            root = "/usr/workspace/trivedi1/vision_data/Living17-Corrupted" 
        elif corruption_name in CBAR_CORRUPTIONS_IMAGENET:
            root = "/usr/workspace/trivedi1/vision_data/Living17-Corrupted-Bar" 
        else:
            print("ERROR ERROR ERROR; undefined corruption. exiting")
            exit()
        ood_dataset = Breeds_C(
            root="{}/{}/{}".format(
                root,corruption_name, severity
            ),
            breeds_name="living17",
            info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
            source=True,  # id-c, we will add ood-c in another iteration.
            target=False,
            split="val",
            transform=transform,
        )
    elif "bird" in args.dataset.lower():
        """
        Get corrupted water birds
        """
        ood_dataset = WILDS_Dataset(
            dataset_name="waterbirds",
            split="test",
            root="/usr/workspace/trivedi1/vision_data/Waterbirds-C",
            meta_selector=waterbirds_selector[args.dataset.lower()],
            transform=transform,
            download=False,
            return_meta=False,
        )

    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        np.random.seed(ss.generate_state(4))

    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=wif,
    )
    return ood_loader


def get_transform(dataset, SELECTED_AUG, use_clip_mean=False,use_vit_mean=False,args=None):
    if dataset == "cifar100":
        num_classes = 100
        crop_size = 32
    elif "cifar10" in dataset.lower():
        num_classes = 10
        crop_size = 32
    elif dataset == "imagenet1K":
        num_classes = 1000
    elif "stl10" in dataset.lower():
        num_classes = 10
    elif "domainnet" in dataset:
        num_classes = 40
        crop_size = 224
    elif "living17" in dataset:
        num_classes = 17
        crop_size = 224
    elif "entity30" in dataset:
        num_classes = 30
        crop_size = 224
    elif "mnist" in dataset.lower():
        num_classes = 10
        crop_size = 224
    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
        exit()
    hparams = {"translate_const": 100, "img_mean": (124, 116, 104)}

    if use_clip_mean:
        normalize = transforms.Normalize(norm_dict["clip_mean"], norm_dict["clip_std"])
    elif use_vit_mean:
        normalize = transforms.Normalize(norm_dict["vits8-dino_mean"], norm_dict["vits8-dino_std"])
    elif args is not None and "101" in args.arch:
        normalize = torch.nn.Identity() #we just convert to tensor for SimCLRv2
    else:
        normalize = transforms.Normalize(
            norm_dict[dataset + "_mean"], norm_dict[dataset + "_std"]
        )
    resize = transforms.Resize(size=(224, 224))

    """
    IMPORTANT: if cutout/mixup/cutmix are selected,
    then, when creating the dataloader, a normalize
    augmentation will be applied in the get_dataloader function 
    to ensure correct behavior
    """

    if SELECTED_AUG == "cutout":
        transform = timm.data.random_erasing.RandomErasing(
            probability=1.0,
            min_area=0.02,
            max_area=1 / 3,
            min_aspect=0.3,
            max_aspect=None,
            mode="const",
            min_count=1,
            max_count=None,
            num_splits=0,
            device=DEVICE,
        )
    elif SELECTED_AUG == "mixup":
        # mixup active if mixup_alpha > 0
        # cutmix active if cutmix_alpha > 0
        transform = timm.data.mixup.Mixup(
            mixup_alpha=1.0,
            cutmix_alpha=0.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.0,
            mode="batch",
            correct_lam=True,
            label_smoothing=0.1,
            num_classes=num_classes,
        )
    elif SELECTED_AUG == "cutmix":
        transform = timm.data.mixup.Mixup(
            mixup_alpha=0.0,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=0.8,
            switch_prob=0.0,
            mode="batch",
            correct_lam=True,
            label_smoothing=0.1,
            num_classes=num_classes,
        )
    elif SELECTED_AUG == "autoaug":
        # no searching code;
        transform = timm.data.auto_augment.auto_augment_transform(
            config_str="original-mstd0.5", hparams=hparams
        )
    elif SELECTED_AUG == "augmix":
        transform = timm.data.auto_augment.augment_and_mix_transform(
            config_str="augmix-m5-w4-d2", hparams=hparams
        )
    elif SELECTED_AUG == "randaug":
        transform = timm.data.auto_augment.rand_augment_transform(
            config_str="rand-m3-n2-mstd0.5", hparams=hparams
        )
    elif SELECTED_AUG == "base":
        transform = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        # else:
        #     transform = transforms.Compose(
        #     [transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, padding=4),
        #     resize,
        #     transforms.ToTensor(),
        #     normalize])

    elif (
        SELECTED_AUG in ["test", "fgsm"]
        or "vat" in SELECTED_AUG
        or "soup" in SELECTED_AUG
    ):
        if dataset.lower() == "mnist":

            class convert_channels:
                def __init__(self, dim=0):
                    self.dim = 0

                def __call__(self, x):
                    x = 0.8 * x
                    return x.repeat(3, 1, 1)

            to_three_channel = convert_channels()
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    to_three_channel,
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
            )
    elif SELECTED_AUG == "pixmix":
        print("Not Implemented yet!")

    if SELECTED_AUG in ["randaug", "augmix", "autoaug"]:
        transform = transforms.Compose(
            [resize, transform, transforms.ToTensor(), normalize]
        )
    return transform


def get_dataloaders(
    args,
    train_aug,
    test_aug,
    train_transform,
    test_transform,
    use_ft=False,
    use_clip_mean=False,
    return_datasets=False,
):
    if train_aug not in ["mixup", "cutmix", "cutout"]:
        # initialize the augmentation directly in the dataset
        if args.dataset == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=train_transform,
            )
        elif args.dataset == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(
                root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=train_transform,
            )
        elif "domainnet" in args.dataset.lower():
            domain_name = args.dataset.split("-")[-1]
            train_dataset = domainnet.DomainNet(
                domain=domain_name,
                split="train",
                root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet",
                transform=train_transform,
            )
        elif "living17" in args.dataset.lower():
            train_dataset = Breeds(
                root="/usr/workspace/trivedi1/vision_data/ImageNet",
                breeds_name="living17",
                info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
                source=True,
                target=False,
                split="train",
                transform=train_transform,
            )
        elif "entity30" in args.dataset.lower():
            train_dataset = Breeds(
                root="/usr/workspace/trivedi1/vision_data/ImageNet",
                breeds_name="entity30",
                info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
                source=True,
                target=False,
                split="train",
                transform=train_transform,
            )
        elif "waterbirds" in args.dataset.lower():
            train_dataset = WILDS_Dataset(
                dataset_name="waterbirds",
                split="train",
                root="/usr/workspace/trivedi1/vision_data/Waterbirds",
                meta_selector=None,
                transform=train_transform,
                download=False,
                return_meta=False,
            )
        else:
            print("***** ERROR ERROR ERROR ******")
            print("Invalid Dataset Selected, Exiting")
            exit()
    else:
        # augmentation will be applied in training loop!
        if use_clip_mean:
            normalize = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_dict["clip_mean"], norm_dict["clip_std"]),
                ]
            )
        elif "vits8-dino" in args.arch:
            normalize = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_dict["vits8-dino_mean"], norm_dict["vits8-dino_std"]),
                ]
            )
        elif "101" in args.arch:
            normalize = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        else:
            normalize = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        norm_dict[args.dataset + "_mean"],
                        norm_dict[args.dataset + "_std"],
                    ),
                ]
            )
        if args.dataset == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root="/p/lustre1/trivedi1/vision_data", train=True, transform=normalize
            )
        elif args.dataset == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(
                root="/p/lustre1/trivedi1/vision_data", train=True, transform=normalize
            )
        elif "domainnet" in args.dataset.lower():
            domain_name = args.dataset.split("-")[-1]
            train_dataset = domainnet.DomainNet(
                domain=domain_name,
                split="train",
                root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet",
                transform=normalize,
            )
        elif "living17" in args.dataset.lower():
            train_dataset = Breeds(
                root="/usr/workspace/trivedi1/vision_data/ImageNet",
                breeds_name="living17",
                info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
                source=True,
                target=False,
                split="train",
                transform=normalize,
            )
        elif "entity30" in args.dataset.lower():
            train_dataset = Breeds(
                root="/usr/workspace/trivedi1/vision_data/ImageNet",
                breeds_name="entity30",
                info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
                source=True,
                target=False,
                split="train",
                transform=normalize,
            )
        elif "waterbirds" in args.dataset.lower():
            train_dataset = WILDS_Dataset(
                dataset_name="waterbirds",
                split="train",
                root="/usr/workspace/trivedi1/vision_data/Waterbirds",
                meta_selector=None,
                transform=normalize,
                download=False,
                return_meta=False,
            )
        else:
            print("***** ERROR ERROR ERROR ******")
            print("Invalid Dataset Selected, Exiting")
            exit()
    """
    Create Test Dataloaders.
    """
    if args.dataset == "cifar10":
        test_dataset = torchvision.datasets.CIFAR10(
            root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=test_transform,
        )
        NUM_CLASSES = 10
    elif args.dataset == "cifar100":
        test_dataset = torchvision.datasets.CIFAR100(
            root="/p/lustre1/trivedi1/vision_data",
            train=False,
            transform=test_transform,
        )
        NUM_CLASSES = 100
    elif "domainnet" in args.dataset.lower():
        domain_name = args.dataset.split("-")[-1]
        test_dataset = domainnet.DomainNet(
            domain=domain_name,
            split="test",
            root="/usr/workspace/wsa/trivedi1/vision_data/DomainNet",
            transform=test_transform,
        )
        NUM_CLASSES = 40
    elif "living17" in args.dataset.lower():
        test_dataset = Breeds(
            root="/usr/workspace/trivedi1/vision_data/ImageNet",
            breeds_name="living17",
            info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
            source=True,
            target=False,
            split="val",
            transform=test_transform,
        )
    elif "entity30" in args.dataset.lower():
        test_dataset = Breeds(
            root="/usr/workspace/trivedi1/vision_data/ImageNet",
            breeds_name="entity30",
            info_dir="/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified",
            source=True,
            target=False,
            split="val",
            transform=test_transform,
        )
    elif "waterbirds" in args.dataset.lower():
        test_dataset = WILDS_Dataset(
            dataset_name="waterbirds",
            split="test",
            root="/usr/workspace/trivedi1/vision_data/Waterbirds",
            meta_selector=None,
            transform=test_transform,
            download=False,
            return_meta=False,
        )

    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
        exit()

    if not return_datasets:
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
        if train_aug in ["mixup", "cutmix"]:
            drop_last = True  # mixup/cutmix needs an even number of batch samples
        else:
            drop_last = False
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=wif,
            drop_last=drop_last,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader
    else:
        # this option is needed for CIFAR-C evaluation
        return train_dataset, test_dataset


def train(
    net,
    train_loader,
    optimizer,
    scheduler,
    transform=None,
    transform_name=None,
    protocol="lp",
    l2sp=False,
):
    """Train for one epoch."""
    if protocol in ["lp", "lp+ft"]:
        # model needs to be eval mode!
        net.eval()
    else:
        net.train()
    loss_ema = 0.0
    if transform_name in ["cutmix", "mixup"]:
        criterion = torch.nn.SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    for _, (images, targets) in enumerate(train_loader):

        optimizer.zero_grad()
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # use cutmix or mixup
        if transform:
            if transform_name in ["cutmix", "mixup"]:
                images, targets = transform(images, target=targets)
            if transform_name == "cutout":
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
    total_loss = 0.0
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
    return np.round(total_loss / len(test_loader), 5), np.round(
        total_correct / len(test_loader.dataset), 5
    )


def freeze_layers_for_lp(model,use_head=False):
    for name, param in model.named_parameters():
        if use_head:
            if "fc" not in name and "head" not in name: 
                param.requires_grad = False
        else:
            if "fc" not in name:
                param.requires_grad = False
    return model


def unfreeze_layers_for_ft(model,use_head=False):
    for param in model.parameters():
        param.requires_grad = True
    return model


def unfreeze_layers_for_lpfrz_ft(model,use_head=False):
    for name, param in model.named_parameters():
        if use_head:
            if "fc" not in name and "head" not in name: 
                param.requires_grad = False
        else:
            if "fc" not in name:
                param.requires_grad = False
        
    return model


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Transfer Learning Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=[
            "cifar10",
            "domainnet-sketch",
            "cifar100",
            "pairedCIFAR",
            "blendedCIFAR",
            "living17",
            "entity30",
        ],
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="stl10",
        choices=[
            "stl10",
            "cifar10.1",
            "domainnet-painting",
            "domainnet-real",
            "domainnet-clipart",
            "domainnet-all",
            "cifar100",
            "pairedSTL",
            "blendedSTL",
            "living17",
            "entity30",
        ],
    )

    parser.add_argument(
        "--arch", type=str, default="resnet50", choices=["resnet50", "clip-RN50", "vits8-dino","r101-1x-sk0"]
    )

    parser.add_argument(
        "--protocol",
        type=str,
        default="lp",
        choices=[
            "lp",
            "ft",
            "lp+ft",
            "lpfrz+ft",
            "sklp",
            "sklp+ft",
            "vatlp",
            "vatlp+ft",
            "voslp",
            "voslp+ft",
            "fgsmlp",
            "fgsmlp+ft",
            "soup-avg-lp",
            "soup-avg-lp+ft",
            "souplp",
            "souplp+ft",
            "udplp",
            "udplp+ft",
        ],
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar",
    )
    """
    Linear Probe Args
    """
    parser.add_argument(
        "--train_aug",
        type=str,
        default="autoaug",
        # choices=['cutout','mixup','cutmix','autoaug','augmix','randaug','base','pixmix','test','vat']
    )
    parser.add_argument("--test_aug", type=str, default="test")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--alpha", type=float, default=1e-1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=0.1, help="Initial learning rate."
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument(
        "--decay", "-wd", type=float, default=0.0005, help="Weight decay (L2 penalty)."
    )
    parser.add_argument(
        "--droprate", default=0.0, type=float, help="Dropout probability"
    )
    parser.add_argument(
        "--save", "-s", type=str, default="./ckpts", help="Folder to save checkpoints."
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default="",
        help="Checkpoint path for resume / test.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Eval only.")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=50,
        help="Training loss print frequency (batches).",
    )
    """
    Finetuning args
    """
    parser.add_argument(
        "--ft_train_aug",
        type=str,
        default="autoaug",
        choices=[
            "cutout",
            "mixup",
            "cutmix",
            "autoaug",
            "augmix",
            "randaug",
            "base",
            "pixmix",
            "test",
        ],
    )
    parser.add_argument("--ft_test_aug", type=str, default="test")
    parser.add_argument("--ft_epochs", type=int, default=20)

    parser.add_argument("--ft_batch_size", type=int, default=64)

    parser.add_argument(
        "--ft_learning-rate",
        "-ft_lr",
        type=float,
        default=0.1,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--l2sp_weight", "-l2", type=float, default=-1, help="Keep LP close to self."
    )

    parser.add_argument(
        "--resume_lp_ckpt",
        type=str,
        default="None",
        help="Optional ckpt to begin LP + FT from.",
    )
    parser.add_argument("--ft_momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument(
        "--ft_decay",
        "-ft_wd",
        type=float,
        default=0.0005,
        help="Weight decay (L2 penalty).",
    )

    """
    VOS
    """
    parser.add_argument("--start_epoch", type=int, default=5)
    parser.add_argument("--sample_number", type=int, default=500)
    parser.add_argument("--select", type=int, default=1)
    parser.add_argument("--sample_from", type=int, default=5000)
    parser.add_argument("--loss_weight", type=float, default=0.01)

    """
    Freezing Batchnorm     
    """
    parser.add_argument("--bn-train-mode", action="store_true")
    parser.add_argument("--bn-eval-mode", dest="train_batchnorm", action="store_false")
    parser.set_defaults(train_batchnorm=True)

    """
    Using Bias in cls?
    """
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--no-bias", dest="use_bias", action="store_false")
    parser.set_defaults(use_bias=True)

    """
    Simplicity Bias
    """
    parser.add_argument(
        "--correlation_strength",
        type=float,
        default=0.89,
        help="How strong the correlation is for blended cifar-mnist",
    )

    """
    Adversarial LP 
    """
    parser.add_argument(
        "--eps",
        type=float,
        default=0.001,
        help="Episilon for LP Training (Hidden Space Adv. Training!)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=10, help="How many steps for PGD Attack"
    )
    parser.add_argument(
        "--num_cls", type=int, default=20, help="How many cls in soup"
    )
    args = parser.parse_args()
    return args


def load_moco_ckpt(model, args):
    ckpt = torch.load(args.pretrained_ckpt)
    new_keys = [
        (k, k.replace("module.encoder_q.", "")) for k in list(ckpt["state_dict"].keys())
    ]
    for old_k, new_k in new_keys:
        ckpt["state_dict"][new_k] = ckpt["state_dict"].pop(old_k)
    incompatible, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("Incompatible Keys: ", incompatible)
    print("Unexpected Keys: ", unexpected)
    return model


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# https://github.com/PatrickHua/SimSiam/blob/main/optimizers/lr_scheduler.py
class LR_Scheduler(object):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        iter_per_epoch,
        constant_predictor_lr=False,
    ):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group["name"] == "predictor":
                param_group["lr"] = self.base_lr
            else:
                lr = param_group["lr"] = self.lr_schedule[self.iter]

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


def get_fixed_dataloaders(args, dataset, train_aug, train_transform):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    if train_aug != "test":
        print("*" * 65)
        print("Calling Evaluation Dataloaders with an Aug: {}!".format(train_aug))
        print("Please make sure this what you really wanted to do!")
        print("*" * 65)
    normalize = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225]),
        ]
    )
    if train_aug not in ["mixup", "cutmix", "cutout"]:
        train_transform_x = train_transform
    else:
        train_transform_x = normalize
    if dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="/p/lustre1/trivedi1/vision_data",
            train=True,
            transform=train_transform_x,
        )
    elif dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="/p/lustre1/trivedi1/vision_data",
            train=True,
            transform=train_transform_x,
        )
    elif dataset == "STL10" or dataset == "stl10":
        train_dataset = torchvision.datasets.STL10(
            root="/p/lustre1/trivedi1/vision_data",
            split="train",
            download=False,
            transform=train_transform_x,
        )

        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
        train_dataset = torch.utils.data.Subset(
            train_dataset, np.where(train_dataset.labels != -1)[0]
        )

    elif dataset.lower() == "cifar10p1":
        train_dataset = CIFAR10p1(
            root="/p/lustre1/trivedi1/vision_data/CIFAR10.1/",
            split="train",
            verision="v6",
            transform=train_transform_x,
        )
    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
        exit()

    """
    Create Test Dataloaders.
    """
    if dataset == "cifar10":
        test_dataset = torchvision.datasets.CIFAR10(
            root="/p/lustre1/trivedi1/vision_data", train=False, transform=normalize
        )
        NUM_CLASSES = 10
    elif dataset == "cifar100":
        test_dataset = torchvision.datasets.CIFAR100(
            root="/p/lustre1/trivedi1/vision_data", train=False, transform=normalize
        )
        NUM_CLASSES = 100
    elif dataset == "STL10" or dataset == "stl10":
        test_dataset = torchvision.datasets.STL10(
            root="/p/lustre1/trivedi1/vision_data",
            split="test",
            download=False,
            transform=normalize,
        )
        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        test_dataset.labels = stl_to_cifar_indices[test_dataset.labels]
        test_dataset = torch.utils.data.Subset(
            test_dataset, np.where(test_dataset.labels != -1)[0]
        )

    elif dataset.lower() == "cifar10p1":
        train_dataset = CIFAR10p1(
            root="/p/lustre1/trivedi1/vision_data/CIFAR10.1/",
            split="test",
            verision="v6",
            transform=normalize,
        )
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
        generator=g,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

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
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample",
    "brownish_noise",
    "checkerboard_cutout",
    "inverse_sparkles",
    "pinch_and_twirl",
    "ripple",
    "circular_motion_blur",
    "lines",
    "sparkles",
    "transverse_chromatic_abberation",
]

CBAR_CORRUPTIONS_SEV = {
    "caustic_refraction": [2.35, 3.2, 4.9, 6.6, 9.15],
    "inverse_sparkles": [1.0, 2.0, 4.0, 9.0, 10.0],
    "sparkles": [1.0, 2.0, 3.0, 5.0, 6.0],
    "perlin_noise": [4.6, 5.2, 5.8, 7.6, 8.8],
    "blue_noise_sample": [0.8, 1.6, 2.4, 4.0, 5.6],
    "plasma_noise": [4.75, 7.0, 8.5, 9.25, 10.0],
    "checkerboard_cutout": [2.0, 3.0, 4.0, 5.0, 6.0],
    "cocentric_sine_waves": [3.0, 5.0, 8.0, 9.0, 10.0],
    "single_frequency_greyscale": [1.0, 1.5, 2.0, 4.5, 5.0],
    "brownish_noise": [1.0, 2.0, 3.0, 4.0, 5.0],
}


def get_calibration_loader(
    args, cal_dataset, corruption=None, severity=None, clean_test_dataset=None
):
    """
    Get different CIFAR10 Calibration datasets
    """
    use_clip_mean = "clip" in args.arch
    use_vit_mean = "vit" in args.arch
    if args.dataset == "cifar10":
        if cal_dataset == "id-c":
            """
            Load cifar10-c
            """
            corruption_path = "/usr/workspace/trivedi1/vision_data/CIFAR-10-C/"
            clean_test_dataset.data = np.load(corruption_path + corruption + ".npy")
            clean_test_dataset.targets = torch.LongTensor(
                np.load(corruption_path + "labels.npy")
            )
            clean_test_loader = torch.utils.data.DataLoader(
                clean_test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        elif cal_dataset == "id-c-bar":
            """
            Load cifar10-c-bar
            """
            corruption_path = "/usr/workspace/trivedi1/vision_data/CIFAR-10-C-BAR/"
            clean_test_dataset.data = np.load(corruption_path + corruption + ".npy")
            clean_test_dataset.targets = torch.LongTensor(
                np.load(corruption_path + "labels.npy")
            )
            clean_test_loader = torch.utils.data.DataLoader(
                clean_test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        elif cal_dataset == "id-clean":
            train_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args
            )
            test_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args
            )
            _, clean_test_loader = get_dataloaders(
                args=args,
                train_aug="test",
                test_aug="test",
                train_transform=train_transform,
                test_transform=test_transform,
                use_clip_mean=use_clip_mean,
            )
        elif cal_dataset == "stl":
            clean_test_loader = get_oodloader(
                args, dataset="stl10", use_clip_mean=use_clip_mean
            )
        elif cal_dataset == "cifar10p1":
            clean_test_loader = get_oodloader(
                args, dataset="cifar10.1", use_clip_mean=use_clip_mean
            )
    elif args.dataset == "cifar100":
        """
        Get CIFAR100 Calibration Datasets
        """
        if cal_dataset == "id-c":
            """
            Load cifar100-c
            """
            corruption_path = "/usr/workspace/trivedi1/vision_data/CIFAR-100-C"
            clean_test_dataset.data = np.load(corruption_path + corruption + ".npy")
            clean_test_dataset.targets = torch.LongTensor(
                np.load(corruption_path + "labels.npy")
            )
            clean_test_loader = torch.utils.data.DataLoader(
                clean_test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            corrs = CORRUPTIONS
        elif cal_dataset == "id-c-bar":
            """
            Load cifar100-c-bar
            """
            corruption_path = "/usr/workspace/trivedi1/vision_data/CIFAR-100-C-BAR"
            clean_test_dataset.data = np.load(corruption_path + corruption + ".npy")
            clean_test_dataset.targets = torch.LongTensor(
                np.load(corruption_path + "labels.npy")
            )
            clean_test_loader = torch.utils.data.DataLoader(
                clean_test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            corrs = CBAR_CORRUPTIONS
        elif cal_dataset == "id-clean":
            train_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args
            )
            test_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args

            )
            _, clean_test_loader = get_dataloaders(
                args=args,
                train_aug="test",
                test_aug="test",
                train_transform=train_transform,
                test_transform=test_transform,
                use_clip_mean=use_clip_mean,
            )
    elif "domain" in args.dataset:
        """
        Get DomainNet Calibration Datasets
        """
        if cal_dataset == "id-c":
            clean_test_loader = get_corrupted_loader(
                args,
                dataset=args.dataset,
                corruption_name=corruption,
                severity=severity,
                use_clip_mean=use_clip_mean,
            )
        elif cal_dataset == "id-c-real":
            clean_test_loader = get_corrupted_loader(
                args,
                dataset="domainnet-real",
                corruption_name=corruption,
                severity=severity,
                use_clip_mean=use_clip_mean,
            )
        elif cal_dataset == "id-clean":
            train_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args
            )
            test_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args
            )
            _, clean_test_loader = get_dataloaders(
                args=args,
                train_aug="test",
                test_aug="test",
                train_transform=train_transform,
                test_transform=test_transform,
                use_clip_mean=use_clip_mean,
            )
        elif "domain" in cal_dataset and "sketch" not in cal_dataset:
            """
            Get other domains ('clipart','painting','real')
            """
            clean_test_loader = get_oodloader(
                args, dataset=cal_dataset, use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean
            )
    elif "living17" in args.dataset:
        """
        Get DomainNet Calibration Datasets
        """
        if cal_dataset == "id-c":
            clean_test_loader = get_corrupted_loader(
                args,
                dataset=args.dataset,
                corruption_name=corruption,
                severity=severity,
                use_clip_mean=use_clip_mean,
            )
        elif cal_dataset == "id-c-bar":
            clean_test_loader = get_corrupted_loader(
                args,
                dataset=args.dataset,
                corruption_name=corruption,
                severity=severity,
                use_clip_mean=use_clip_mean,
            )
        elif cal_dataset == "id-clean":
            train_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args
            )
            test_transform = get_transform(
                dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean,
                use_vit_mean=use_vit_mean,
                args=args
            )
            _, clean_test_loader = get_dataloaders(
                args=args,
                train_aug="test",
                test_aug="test",
                train_transform=train_transform,
                test_transform=test_transform,
                use_clip_mean=use_clip_mean,
            )
        elif "ood" in cal_dataset:
            """
            Get Subpopulation Shift Living17
            """
            clean_test_loader = get_oodloader(
                args, dataset="living17", use_clip_mean=use_clip_mean  # intentional!
            )
        else:
            print("LOADER IS NOT IMPLEMENTED")
    return clean_test_loader

#Code from: https://github.com/mpagli/Uncertainty-Driven-Perturbations/blob/59a915bc98da0532bce51b84dcfc59d54ee648ac/src/utils.py#L79
def get_udp_perturbation(model, X, y, eps, alpha, attack_iters, rs=False, clamp_val=None, use_alpha_scheduler=False, sample_iters='none'):
    # y is not used, just here to keep the interface
    
    delta = torch.zeros_like(X).to(X.device)
        
    if use_alpha_scheduler:
        alpha_scheduler = lambda t: np.interp([t], [0, attack_iters // 2, attack_iters], [alpha, max(eps/2, alpha), alpha])[0]

    if rs:
        delta.uniform_(-eps, eps)

    if sample_iters == 'uniform':
        shape = [delta.shape[0]] + [1] * (len(delta.shape)-1)
        sampled_iters = torch.randint(1,attack_iters+1,shape).expand_as(delta).to(X.device)

    delta.requires_grad = True

    for itr in range(attack_iters):

        if clamp_val is not None:
            X_ = torch.clamp(X + delta, clamp_val[0], clamp_val[1])
        else:
            X_ = X + delta

        output = model(X_)
        p = torch.softmax(output, dim=1)
        entropy = - (p * p.log()).sum(dim=1)
        entropy.mean().backward()
        grad = delta.grad.detach().sign()
        if sample_iters != 'none':
            grad[sampled_iters <= itr] = 0.0
        if use_alpha_scheduler:
            delta.data = delta + alpha_scheduler(itr+1) * grad
        else:
            delta.data = delta + alpha * grad
        if clamp_val is not None:
            delta.data = torch.clamp(X + delta.data, clamp_val[0], clamp_val[1]) - X
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.grad.zero_()

    return delta.detach()
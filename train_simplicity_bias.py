from asyncio import protocols
import os
import time
from lps_utils import create_sparse_clfs, test_soup_reps
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
from clip_model import ClipModel
from torch.utils.data.distributed import DistributedSampler
from vat_loss import VATLoss
from paired_cifar_mnist import pairedCIFARMNIST, pairedSTLMNIST
from blended_cifar_mnist import blendedCIFARMNIST, blendedSTLMNIST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
Wrapped into function so we 
can call it for fine-tuning too.
"""
PREFIX = "/usr/workspace/trivedi1/simplicity_experiments_aaai/"


def train_loop(
    args,
    protocol,
    save_name,
    log_path,
    net,
    optimizer,
    scheduler,
    start_epoch,
    end_epoch,
    train_loader,
    test_loader,
    train_aug,
    train_transform,
):

    use_clip_mean = "clip" in args.arch
    best_acc = 0
    weight_dict_initial, _ = get_param_weights_counts(net, detach=True)
    # if 'ft' in protocol:
    """
    Get spuriously correlated OOD datasets.
    """
    ood_dataset = blendedSTLMNIST(train=False, transform=None, use_paired=True)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    random_ood_dataset = blendedSTLMNIST(
        train=False, transform=None, randomized=True, use_paired=True
    )
    random_ood_loader = torch.utils.data.DataLoader(
        random_ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    random_id_test = blendedCIFARMNIST(train=False, randomized=True, use_paired=True)
    random_id_loader = torch.utils.data.DataLoader(
        random_id_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("=> Beginning training from epoch:", start_epoch + 1)
    l2sp_loss = -1
    if args.l2sp_weight != -1:
        print("=> Using l2sp weight: ", args.l2sp_weight)
        l2sp_loss = 0
    if train_aug in ["cutmix", "mixup", "cutout"]:
        transform = train_transform
    else:
        transform = None
    if train_aug in ["cutmix", "mixup"]:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if (
        protocol in ["lp"] or not args.train_batchnorm
    ):  # note: protocol re-specified in the main function as lp or ft ONLY.
        print("****** Freezing Batchnorm Parameters ******")
    else:
        print("****** Updating Batchnorm Parameters ****")

    stat_dict = {}
    stat_dict["spur_id_acc"] = []
    stat_dict["rand_id_acc"] = []
    stat_dict["spur_ood_acc"] = []
    stat_dict["rand_ood_acc"] = []
    stat_dict["complex_feat_acc"] = []
    stat_dict["simple_feat_acc"] = []
    for epoch in range(start_epoch, end_epoch):
        begin_time = time.time()
        if (
            protocol in ["lp"] or not args.train_batchnorm
        ):  # note: protocol re-specified in the main function as lp or ft ONLY.
            net.eval()
        else:
            net.train()
        loss_ema = 0.0
        for _, (images, targets) in tqdm.tqdm(enumerate(train_loader), disable=True):
            optimizer.zero_grad()
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # use cutmix or mixup
            if transform:
                if train_aug in ["cutmix", "mixup"]:
                    images, targets = transform(images, target=targets)
                if train_aug == "cutout":
                    images = transform(images)
            logits = net(images)
            loss = criterion(logits, targets)
            if args.l2sp_weight != -1:
                weight_dict, _ = get_param_weights_counts(net, detach=False)
                l2sp_loss = args.l2sp_weight * get_l2_dist(
                    weight_dict_initial, weight_dict
                )
                loss += l2sp_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1

        test_loss, spur_id_acc = test(net, test_loader)

        ood_acc = -1
        complex_feat_acc = -1
        simple_feat_acc = -1
        _, spur_ood_acc = test(net, ood_loader)
        _, rand_ood_acc = test(net, random_ood_loader)

        ### going to load the wa
        # _, complex_feat_acc = test(net, complex_feat_loader)

        _, rand_id_acc = test(net, random_id_loader)
        # random_id_loader.dataset.return_simple_label=True
        # _, simple_feat_acc = test(net, random_id_loader)
        # random_id_loader.dataset.return_simple_label=False
        print(
            "Epoch {epoch:3d} | Time {time_s:5d} | Train Loss {train_loss:.4f} | Test Loss {test_loss:.3f} |"
            " S-Test Error {spur_id:.2f} | R-Test Error {rand_id:.2f} | S-OOD Error {spur_ood:.2f} | R-OOD Error {rand_ood:.2f} | Simp-Error {simple:.2f} | Comp-Error {complex:.2f}".format(
                epoch=(epoch + 1),
                time_s=int(time.time() - begin_time),
                train_loss=loss_ema,
                test_loss=test_loss,
                spur_id=100 - 100.0 * spur_id_acc,
                rand_id=100 - 100.0 * rand_id_acc,
                spur_ood=100 - 100.0 * spur_ood_acc,
                rand_ood=100 - 100.0 * rand_ood_acc,
                simple=100 - 100.0 * simple_feat_acc,
                complex=100 - 100.0 * complex_feat_acc,
            )
        )

        """
        Update stat dict
        """
        stat_dict["spur_id_acc"].append(spur_id_acc)
        stat_dict["rand_id_acc"].append(rand_id_acc)
        stat_dict["spur_ood_acc"].append(spur_ood_acc)
        stat_dict["rand_ood_acc"].append(rand_ood_acc)
        stat_dict["complex_feat_acc"].append(complex_feat_acc)
        stat_dict["simple_feat_acc"].append(simple_feat_acc)
        """
        save best checkpoint?
        """
        is_best = spur_id_acc > best_acc
        best_acc = max(spur_id_acc, best_acc)
        # if is_best:
        #     checkpoint = {
        #     'epoch': epoch,
        #     'dataset': args.dataset,
        #     'model': args.arch,
        #     'state_dict': net.state_dict(),
        #     'best_acc': best_acc,
        #     'optimizer': optimizer.state_dict(),
        #     'protocol':args.protocol
        #     }
        #     save_path = os.path.join(args.save, save_name + "_" + args.protocol +'_model_best.pth.tar')
        #     torch.save(checkpoint, save_path)

    checkpoint = {
        "epoch": epoch,
        "dataset": args.dataset,
        "model": args.arch,
        "state_dict": net.state_dict(),
        "best_acc": best_acc,
        "optimizer": optimizer.state_dict(),
        "protocol": args.protocol,
        "stat_dict": stat_dict,
    }
    return net, checkpoint


def extract_features(args, model, loader, train_aug, train_transform):
    if train_aug in ["cutmix", "mixup", "cutout"]:
        transform = train_transform
    else:
        transform = None
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm.tqdm(loader, disable=True):
            data = data.cuda()
            if transform:
                if train_aug in ["cutmix", "mixup"]:
                    images, targets = transform(images, target=targets)
                if train_aug == "cutout":
                    images = transform(images)
            if args.arch == "resnet50":
                # using a timm model. called the
                reps = torch.nn.functional.adaptive_avg_pool2d(
                    model.module.forward_features(data), 1
                )
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
            if "clip" in args.arch:
                # using a timm model. called the
                reps = model.module.get_features(data)
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
    # features = torch.nn.functional.adaptive_avg_pool2d(np.squeeze(np.concatenate(features)),1)
    features = np.squeeze(np.concatenate(features))
    labels = np.concatenate(labels)
    return features, labels


def linear_probe_vat(args, net, train_loader, test_loader, train_aug, train_transform):
    net.eval()
    train_features, train_labels = extract_features(
        args,
        model=net,
        loader=train_loader,
        train_aug=train_aug,
        train_transform=train_transform,
    )
    test_features, test_labels = extract_features(
        args, model=net, loader=test_loader, train_aug="test", train_transform=None
    )

    """Get OOD Loaders"""
    ood_dataset = blendedSTLMNIST(train=False, transform=None,use_paired=True)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    random_ood_dataset = blendedSTLMNIST(train=False, transform=None, randomized=True,use_paired=True)
    random_ood_loader = torch.utils.data.DataLoader(
        random_ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ood_features, ood_labels = extract_features(
        args,
        model=net,
        loader=ood_loader,
        train_aug="test",
        train_transform=train_transform,
    )
    rand_ood_features, rand_ood_labels = extract_features(
        args,
        model=net,
        loader=random_ood_loader,
        train_aug="test",
        train_transform=None,
    )

    print(train_features.shape)

    rep_train_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(train_features), torch.Tensor(train_labels).long()
    )
    rep_train_dataloader = torch.utils.data.DataLoader(
        rep_train_dataset, batch_size=args.batch_size, shuffle=True
    )

    rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(test_features), torch.Tensor(test_labels).long()
    )
    rep_test_dataloader = torch.utils.data.DataLoader(
        rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(ood_features), torch.Tensor(ood_labels).long()
    )
    ood_rep_test_dataloader = torch.utils.data.DataLoader(
        ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    rand_ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(rand_ood_features), torch.Tensor(rand_ood_labels).long()
    )
    rand_ood_rep_test_dataloader = torch.utils.data.DataLoader(
        rand_ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    """
    Create a linear probe layer.
    We will attach it separately back.
    """

    fc = torch.nn.Linear(train_features.shape[1], NUM_CLASSES_DICT[args.dataset]).to(
        DEVICE
    )
    optimizer = torch.optim.SGD(
        fc.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True,
    )

    scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs=0,
        warmup_lr=0 * args.batch_size / 256,
        num_epochs=args.epochs,
        base_lr=args.learning_rate * args.batch_size / 256,
        final_lr=1e-5 * args.batch_size / 256,
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=False,
    )
    ALPHA = args.alpha
    criterion = torch.nn.CrossEntropyLoss()
    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
    for epochs in range(args.epochs):
        vat_avg = 0
        consistency_loss_avg = 0
        loss_avg = 0
        for batch_idx, (data, target) in enumerate(rep_train_dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            # LDS should be calculated before the forward for cross entropy
            lds = vat_loss(fc, data)
            output = fc(data)
            consistency_loss = criterion(output, target)
            loss = consistency_loss + ALPHA * lds
            loss.backward()
            optimizer.step()
            scheduler.step()

            vat_avg += ALPHA * lds
            consistency_loss_avg += consistency_loss
            loss_avg += loss
        _, train_acc = test(fc, rep_train_dataloader)
        _, val_acc = test(fc, rep_test_dataloader)
        _, ood_acc = test(fc, ood_rep_test_dataloader)
        _, rand_ood_acc = test(fc, rand_ood_rep_test_dataloader)

        vat_avg /= len(rep_train_dataloader)
        consistency_loss_avg /= len(rep_train_dataloader)
        loss_avg /= len(rep_train_dataloader)
        print(
            "Epoch: {4} -- VAT: {0:.4f} -- Con: {1:.4f} -- Tot.:{2:.4f} -- Train Error: {3:.4f} -- Test Error: {5:.4f} -- OOD Error: {6:.4f} -- Rand OOD Error: {7:.4f}".format(
                vat_avg,
                consistency_loss_avg,
                loss_avg,
                100 - 100.0 * train_acc,
                epochs,
                100 - 100.0 * val_acc,
                100 - 100.0 * ood_acc,
                100 - 100.0 * rand_ood_acc,
            )
        )
    # Set the classifier weights
    net.module.fc = fc

    """
    Compute acc. using forward passes.
    """
    _, acc = test(net=net, test_loader=test_loader)
    print("Completed VAT Training: {0:.3f}".format(acc))
    checkpoint = {
        "lp_epoch": args.epochs,
        "dataset": args.dataset,
        "model": args.arch,
        "state_dict": net.state_dict(),
        "best_acc": acc,
        "protocol": "vatlp",
    }
    return net, checkpoint

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

def linear_probe_udp(args, net, train_loader, test_loader, train_aug, train_transform):

    net.eval()
    """
    Get in-distribution feature representations
    """
    print("Creating Tensor Datasets...")
    train_features, train_labels = extract_features(
        args,
        model=net,
        loader=train_loader,
        train_aug=train_aug,
        train_transform=train_transform,
    )
    test_features, test_labels = extract_features(
        args, model=net, loader=test_loader, train_aug="test", train_transform=None
    )

    """Get OOD Loaders"""
    ood_dataset = blendedSTLMNIST(train=False, transform=None,use_paired=True,randomized=False)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ood_features, ood_labels = extract_features(
        args,
        model=net,
        loader=ood_loader,
        train_aug="test",
        train_transform=train_transform,
    ) 
    
    random_ood_dataset = blendedSTLMNIST(train=False, transform=None, randomized=True,use_paired=True)
    random_ood_loader = torch.utils.data.DataLoader(
        random_ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    rand_ood_features, rand_ood_labels = extract_features(
        args,
        model=net,
        loader=random_ood_loader,
        train_aug="test",
        train_transform=None,
    )

    """
    Create Representation Data Loaders
    """
    print(train_features.shape)
    print(train_features.mean(), train_features.max(),train_features.min())
    
    rep_train_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(train_features), torch.Tensor(train_labels).long()
    )
    rep_train_dataloader = torch.utils.data.DataLoader(
        rep_train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(test_features), torch.Tensor(test_labels).long()
    )
    rep_test_dataloader = torch.utils.data.DataLoader(
        rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(ood_features), torch.Tensor(ood_labels).long()
    )
    ood_rep_test_dataloader = torch.utils.data.DataLoader(
        ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    rand_ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(rand_ood_features), torch.Tensor(rand_ood_labels).long()
    )
    rand_ood_rep_test_dataloader = torch.utils.data.DataLoader(
        rand_ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    """
    Create a linear probe layer.
    We will attach it separately back.
    """

    fc = torch.nn.Linear(train_features.shape[1], NUM_CLASSES_DICT[args.dataset]).to(
        DEVICE
    )
    
    optimizer = torch.optim.SGD(
        fc.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True,
    )

    scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs=0,
        warmup_lr=0 * args.batch_size / 256,
        num_epochs=args.epochs,
        base_lr=args.learning_rate * args.batch_size / 256,
        final_lr=1e-5 * args.batch_size / 256,
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=False,
    )
    criterion = torch.nn.CrossEntropyLoss()
    ood_acc, rand_ood_acc = -1, -1 
    for epochs in range(args.epochs):
        loss_avg = 0
        loss_adv_avg = 0
        loss_clean_avg = 0
        for batch_idx, (data, target) in enumerate(rep_train_dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            delta = get_udp_perturbation(model=fc, 
                X=data, 
                y=target, 
                eps=args.eps, 
                alpha=args.alpha, 
                attack_iters=args.num_steps, 
                rs=False, 
                clamp_val=None, 
                use_alpha_scheduler=False, 
                sample_iters='none')
            optimizer.zero_grad()

            #option for a burn-in period
            clean_output = fc(data)
            adv_output = fc(data+delta)
            loss_clean = criterion(clean_output, target)
            loss_adv =  args.loss_weight * criterion(adv_output,target)
            #print(loss_clean, loss_adv)
            # if epochs > 5: 
            loss = loss_clean + loss_adv 
            # else:
            #     loss = loss_clean
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += loss
            loss_adv_avg += loss_adv
            loss_clean_avg += loss_clean 
        _, train_acc = test(fc, rep_train_dataloader)
        _, val_acc = test(fc, rep_test_dataloader)
        _, ood_acc = test(fc, ood_rep_test_dataloader)
        _, rand_ood_acc = test(fc, rand_ood_rep_test_dataloader)

        loss_avg /= len(rep_train_dataloader)
        loss_clean_avg /= len(rep_train_dataloader)
        loss_adv_avg /= len(rep_train_dataloader)
        print(
            "UDP, Epoch: {0} -- Loss: {1:.3f} -- Clean: {6:.3f} -- UDP: {7:.3f} -- Train Error: {2:.4f} -- Test Error: {3:.4f} -- OOD Error: {4:.4f} -- Rand OOD Error: {5:.4f} -- ".format(
                epochs,
                loss_avg,
                100 - 100.0 * train_acc,
                100 - 100.0 * val_acc,
                100 - 100.0 * ood_acc,
                100 - 100.0 * rand_ood_acc,
                loss_clean_avg,
                loss_adv_avg
            )
        )
    # Set the classifier weights
    net.module.fc = fc

    """
    Compute acc. using forward passes.
    """
    spur_ood_acc,rand_ood_acc = -1, -1
    _, spur_test_acc = test(net=net, test_loader=test_loader)
    # _, spur_ood_acc = test(net=net, test_loader=ood_loader)
    # _, rand_ood_acc = test(net=net, test_loader=random_ood_loader)

    print("Completed UDP Training: S-test: {0:.3f} -- S-OOD: {1:.3f} -- R-OOD: {2:.3f}".format(spur_test_acc,spur_ood_acc,rand_ood_acc))
    checkpoint = {
        "lp_epoch": args.epochs,
        "dataset": args.dataset,
        "model": args.arch,
        "state_dict": net.state_dict(),
        "best_acc": spur_test_acc,
        "protocol": "vatlp",
    }
    return net, checkpoint

def linear_probe_vanilla(args, net, train_loader, test_loader, train_aug, train_transform):

    net.eval()
    """
    Get in-distribution feature representations
    """
    print("Creating Tensor Datasets...")
    train_features, train_labels = extract_features(
        args,
        model=net,
        loader=train_loader,
        train_aug=train_aug,
        train_transform=train_transform,
    )
    test_features, test_labels = extract_features(
        args, model=net, loader=test_loader, train_aug="test", train_transform=None
    )

    """Get OOD Loaders"""
    ood_dataset = blendedSTLMNIST(train=False, transform=None,use_paired=True,randomized=False)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ood_features, ood_labels = extract_features(
        args,
        model=net,
        loader=ood_loader,
        train_aug="test",
        train_transform=train_transform,
    ) 
    
    random_ood_dataset = blendedSTLMNIST(train=False, transform=None, randomized=True,use_paired=True)
    random_ood_loader = torch.utils.data.DataLoader(
        random_ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    rand_ood_features, rand_ood_labels = extract_features(
        args,
        model=net,
        loader=random_ood_loader,
        train_aug="test",
        train_transform=None,
    )

    """
    Create Representation Data Loaders
    """
    print(train_features.shape)
    print(train_features.mean())
    
    rep_train_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(train_features), torch.Tensor(train_labels).long()
    )
    rep_train_dataloader = torch.utils.data.DataLoader(
        rep_train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(test_features), torch.Tensor(test_labels).long()
    )
    rep_test_dataloader = torch.utils.data.DataLoader(
        rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(ood_features), torch.Tensor(ood_labels).long()
    )
    ood_rep_test_dataloader = torch.utils.data.DataLoader(
        ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    rand_ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(rand_ood_features), torch.Tensor(rand_ood_labels).long()
    )
    rand_ood_rep_test_dataloader = torch.utils.data.DataLoader(
        rand_ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    """
    Create a linear probe layer.
    We will attach it separately back.
    """

    fc = torch.nn.Linear(train_features.shape[1], NUM_CLASSES_DICT[args.dataset]).to(
        DEVICE
    )
    
    optimizer = torch.optim.SGD(
        fc.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True,
    )

    scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs=0,
        warmup_lr=0 * args.batch_size / 256,
        num_epochs=args.epochs,
        base_lr=args.learning_rate * args.batch_size / 256,
        final_lr=1e-5 * args.batch_size / 256,
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=False,
    )
    criterion = torch.nn.CrossEntropyLoss()
    ood_acc, rand_ood_acc = -1, -1 
    for epochs in range(args.epochs):
        loss_avg = 0
        loss_adv_avg = 0
        loss_clean_avg = 0
        for batch_idx, (data, target) in enumerate(rep_train_dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            clean_output = fc(data)
            loss = criterion(clean_output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += loss
        _, train_acc = test(fc, rep_train_dataloader)
        _, val_acc = test(fc, rep_test_dataloader)
        _, ood_acc = test(fc, ood_rep_test_dataloader)
        _, rand_ood_acc = test(fc, rand_ood_rep_test_dataloader)

        loss_avg /= len(rep_train_dataloader)
        print(
            "Vanilla, Epoch: {0} -- Loss: {1:.3f} -- Train Error: {2:.4f} -- Test Error: {3:.4f} -- OOD Error: {4:.4f} -- Rand OOD Error: {5:.4f} -- ".format(
                epochs,
                loss_avg,
                100 - 100.0 * train_acc,
                100 - 100.0 * val_acc,
                100 - 100.0 * ood_acc,
                100 - 100.0 * rand_ood_acc,
            )
        )
    # Set the classifier weights
    net.module.fc = fc

    """
    Compute acc. using forward passes.
    """
    spur_ood_acc,rand_ood_acc = -1, -1
    _, spur_test_acc = test(net=net, test_loader=test_loader)
    # _, spur_ood_acc = test(net=net, test_loader=ood_loader)
    # _, rand_ood_acc = test(net=net, test_loader=random_ood_loader)

    print("Completed UDP Training: S-test: {0:.3f} -- S-OOD: {1:.3f} -- R-OOD: {2:.3f}".format(spur_test_acc,spur_ood_acc,rand_ood_acc))
    checkpoint = {
        "lp_epoch": args.epochs,
        "dataset": args.dataset,
        "model": args.arch,
        "state_dict": net.state_dict(),
        "best_acc": spur_test_acc,
        "protocol": "vatlp",
    }
    return net, checkpoint

def linear_probe_soup(args, net, train_loader,test_loader, train_aug, train_transform):
    net.eval()
    print("=> Extracting Features...")
    train_features, train_labels = extract_features(args,model=net, loader=train_loader,train_aug=train_aug,train_transform=train_transform)
    test_features, test_labels = extract_features(args,model=net, loader=test_loader,train_aug='test',train_transform=None)
    print(train_features.shape)

    rep_train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_features),torch.Tensor(train_labels).long())
    rep_train_dataloader = torch.utils.data.DataLoader(rep_train_dataset,batch_size=args.batch_size,shuffle=True) 
    
    rep_test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_features),torch.Tensor(test_labels).long())
    rep_test_dataloader = torch.utils.data.DataLoader(rep_test_dataset,batch_size=args.batch_size,shuffle=True) 

    """Get OOD Loaders"""
    ood_dataset = blendedSTLMNIST(train=False, transform=None,use_paired=True,randomized=False)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ood_features, ood_labels = extract_features(
        args,
        model=net,
        loader=ood_loader,
        train_aug="test",
        train_transform=train_transform,
    ) 
    
    random_ood_dataset = blendedSTLMNIST(train=False, transform=None, randomized=True,use_paired=True)
    random_ood_loader = torch.utils.data.DataLoader(
        random_ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    rand_ood_features, rand_ood_labels = extract_features(
        args,
        model=net,
        loader=random_ood_loader,
        train_aug="test",
        train_transform=None,
    )
    ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(ood_features), torch.Tensor(ood_labels).long()
    )
    ood_rep_test_dataloader = torch.utils.data.DataLoader(
        ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    rand_ood_rep_test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(rand_ood_features), torch.Tensor(rand_ood_labels).long()
    )
    rand_ood_rep_test_dataloader = torch.utils.data.DataLoader(
        rand_ood_rep_test_dataset, batch_size=args.batch_size, shuffle=True
    )

    """
    Create a set of sparse, orthogonal linear probes layer.
    We will attach it separately back.
    """
    classifier_pool = torch.nn.ModuleList([torch.nn.Linear(train_features.shape[1],NUM_CLASSES_DICT[args.dataset],bias=args.use_bias).to(DEVICE) for _ in range(args.num_cls)])
    with torch.no_grad():
        print("Pre Sparse Classifier Inits: ",[np.round(c.weight.data.norm().data.item(),4) for c in classifier_pool])
    classifier_pool = create_sparse_clfs(classifier_pool=classifier_pool)
    with torch.no_grad():
        print("Sparsified: ",[np.round(c.weight.data.norm().data.item(),4) for c in classifier_pool])
    net.module.classifier_pool = classifier_pool
    optimizer = torch.optim.SGD(
        net.module.classifier_pool.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs=0, warmup_lr = 0*args.batch_size/256, 
        num_epochs=args.epochs, base_lr=args.learning_rate*args.batch_size/256, 
        final_lr =1e-5 *args.batch_size/256, 
        iter_per_epoch= len(train_loader),
        constant_predictor_lr=False
    )
    criterion = torch.nn.CrossEntropyLoss()
    for epochs in range(args.epochs):
        loss_avg = 0
        avg_val = 1 / len(net.module.classifier_pool)
        for batch_idx, (data, target) in enumerate(rep_train_dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = 0
            for cls in net.module.classifier_pool:
                # loss += avg_val * criterion(cls(data),target) 
                loss = loss + criterion(cls(data),target) 
            loss = loss / len(net.module.classifier_pool)  
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += loss 
        _,train_acc = test_soup_reps(net, rep_train_dataloader,args) 
        _,test_acc = test_soup_reps(net, rep_test_dataloader,args) 
        _,ood_acc = test_soup_reps(net, ood_rep_test_dataloader,args) 
        _, rand_ood_acc = test_soup_reps(net, rand_ood_rep_test_dataloader,args) 

        print(
            "SOUP, Epoch: {0} -- Loss: {1:.3f} -- Train Error: {2:.4f} -- Test Error: {3:.4f} -- OOD Error: {4:.4f} -- Rand OOD Error: {5:.4f} -- ".format(
                epochs,
                loss_avg,
                100 - 100.0 * train_acc,
                100 - 100.0 * test_acc,
                100 - 100.0 * ood_acc,
                100 - 100.0 * rand_ood_acc,
            )
        ) 
    """
    After training is complete, we are going to set the fully connected layer.
    Right now, I am taking the average soup. But, I'll save the classifier pool
    in the checkpoint so that other soup recipes can be applied.
    """  
    #make average soup.
    print("=> Making Avg. Soup")
    net.eval()
    with torch.no_grad():
        sum_weight = torch.zeros_like(net.module.classifier_pool[0].weight.data)
        if args.use_bias:
            bias_weight = torch.zeros_like(net.module.classifier_pool[0].bias.data)

        for cls in net.module.classifier_pool:
            sum_weight += cls.weight.data.detach().clone()
            if args.use_bias:
                bias_weight += cls.bias.data.detach().clone()
        
        sum_weight = torch.div(sum_weight, len(net.module.classifier_pool))
        if args.use_bias:
            bias_weight= torch.div(bias_weight, len(net.module.classifier_pool))

        avg_cls = torch.nn.Linear(in_features=sum_weight.shape[0],out_features=sum_weight.shape[1],bias=args.use_bias)
        avg_cls.weight = torch.nn.Parameter(sum_weight)
        if args.use_bias:
            avg_cls.bias = torch.nn.Parameter(bias_weight)
        avg_cls.requires_grad = False  
    """
    Make Greedy Soup
    """
    # print("=> Making Greedy Soup! ")
    # acc_list = []
    # for classifier in net.module.classifier_pool:
    #     net.module.fc = classifier
    #     _, acc = test(net=net,test_loader=test_loader)
    #     acc_list.append(np.round(acc,4))
    # print("Accs: ",acc_list)
    # idx = np.argsort(-1.0 * np.array(acc_list))
    # greedy_cls = torch.nn.Linear(net.module.classifier_pool[idx[0]].weight.shape[0],classifier.weight.shape[1],bias=args.use_bias)
    # greedy_cls.weight =  torch.nn.Parameter(net.module.classifier_pool[idx[0]].weight.data.detach().clone(),requires_grad=False) 
    # if args.use_bias:
    #     greedy_cls.bias =  torch.nn.Parameter(net.module.classifier_pool[idx[0]].bias.data.detach().clone(),requires_grad=False) 
    
    # temp_cls = torch.nn.Linear(net.module.classifier_pool[idx[0]].weight.shape[0],classifier.weight.shape[1],bias=args.use_bias)
    # temp_cls.weight =  torch.nn.Parameter(net.module.classifier_pool[idx[0]].weight.data.detach().clone(),requires_grad=False) 
    # if args.use_bias:
    #     temp_cls.bias=  torch.nn.Parameter(net.module.classifier_pool[idx[0]].bias.data.detach().clone(),requires_grad=False) 
    # curr_acc = acc_list[idx[0]]
    # for enum,i in enumerate(idx[1:]): 
    #     cls = net.module.classifier_pool[i]
        
    #     """
    #     We are going to combine classifier_pool!
    #     """
    #     temp_cls.weight = torch.nn.Parameter(0.5 * (greedy_cls.weight.detach().clone() + cls.weight.detach().clone()),requires_grad=False)
    #     if args.use_bias:
    #         temp_cls.bias = torch.nn.Parameter(0.5 * (greedy_cls.bias.detach().clone() + cls.bias.detach().clone()),requires_grad=False)
    #     net.module.fc = temp_cls.cuda()
    #     _, acc = test(net=net,test_loader=test_loader)
    #     if acc > curr_acc:
    #         print("Adding {0} to soup -- {1:.3f}!".format(enum,curr_acc))
    #         curr_acc = acc
    #         greedy_cls.weight = torch.nn.Parameter(temp_cls.weight.clone().detach(),requires_grad=False)
    #         if args.use_bias:
    #             greedy_cls.bias = torch.nn.Parameter(temp_cls.bias.clone().detach(),requires_grad=False)
    # net.soup_classifier = greedy_cls 
    # net.module.fc = greedy_cls.cuda() 
    # _,soup_acc = test(net=net,test_loader=test_loader) 
    # print('=> Greedy Soup Acc: ',soup_acc)
    
    """
    Compute acc. using forward passes.
    """
    net.module.fc = avg_cls.cuda()
    _,acc = test(net=net,test_loader=test_loader)
    print("=> Final, (Avg) Soup Acc. : {0:.4f}".format(acc))
    checkpoint = {
        'lp_epoch': args.epochs,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'classifier_pool': net.module.classifier_pool,
        'best_acc': acc,
        'protocol':'soup-avg-lp'
    }
    return net, checkpoint

def main():
    args = arg_parser()
    for arg in sorted(vars(args)):
        print("=> ", arg, getattr(args, arg))
    print("=> Device: ", DEVICE)
    print("=> Num GPUS: ", torch.cuda.device_count())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception("%s is not a dir" % args.save)

    if args.pretrained_ckpt == "None":
        if args.arch.lower() == "resnet50":
            print("\t*** Using Default Pretrained Model!!")
            net = timm.create_model(args.arch, pretrained=True)
        if "clip" in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ", encoder_type)
            print("\t => Using Scratch Clip Encoder!")
            net = ClipModel(model_name=encoder_type, scratch=True)
    else:
        if args.arch.lower() == "resnet50":
            net = timm.create_model(args.arch, pretrained=False)
            net = load_moco_ckpt(model=net, args=args)
        if "clip" in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ", encoder_type)
            print("\t => Using Default Clip Ckpt!!")
            net = ClipModel(model_name=encoder_type, scratch=False)
    use_clip_mean = "clip" in args.arch
    if "vat" in args.protocol:
        lp_aug_name = "vat-{}".format(args.alpha)
    elif "udp" in args.protocol:
        lp_aug_name = "udp-{}-{}-{}".format(args.eps, args.alpha,args.loss_weight)
    elif "soup" in args.protocol:
        lp_aug_name = "soup-{}".format(args.num_cls)
    else:
        lp_aug_name = args.train_aug
    dataset_name = "{}-{}".format(args.dataset, args.correlation_strength)
    save_name = (
        dataset_name
        + "_"
        + args.arch
        + "_"
        + args.protocol
        + "_"
        + lp_aug_name
        + "_"
        + args.ft_train_aug
        + "_"
        + str(args.epochs)
        + "_"
        + str(args.learning_rate)
        + "_"
        + str(args.decay)
        + "_"
        + str(args.ft_epochs)
        + "_"
        + str(args.ft_learning_rate)
        + "_"
        + str(args.ft_decay)
        + "_"
        + str(args.l2sp_weight)
        + "_"
        + str(args.seed)
        + "_"
        + str(args.train_batchnorm)
    )

    print("******************************")
    print(save_name)
    print("******************************")

    """
    Throw away classifier.
    Create new classifier with number of classes.
    """
    net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
    print("Reset Classifer: ", net.get_classifier())
    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    torch.backends.cudnn.benchmark = True

    """
    Performing Linear Probe Training!
    """
    lp_train_acc, lp_test_acc, lp_train_loss, lp_test_loss = -1, -1, -1, -1
    ft_train_acc, ft_test_acc, ft_train_loss, ft_test_loss = -1, -1, -1, -1
    if args.protocol in ["lp", "lp+ft", "vatlp+ft", "vatlp","udplp", "udplp+ft","souplp","souplp+ft"]:

        log_path = os.path.join(
            "{}/logs".format(PREFIX), "lp+" + save_name + "_training_log.csv"
        )

        """
        Select Augmentation Scheme.
        """
        train_dataset = blendedCIFARMNIST(
            train=True,
            randomized=False,
            correlation_strength=args.correlation_strength,
        )
        test_dataset = blendedCIFARMNIST(
            train=False,
            randomized=False,
            correlation_strength=args.correlation_strength,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        NUM_CLASSES = NUM_CLASSES_DICT[args.dataset]
        print("=> Num Classes: ", NUM_CLASSES)
        print("=> Train: ", len(train_loader.dataset))
        print("=> Test: ", len(test_loader.dataset))

        # test_loss, test_acc = test(net, test_loader)
        test_loss, test_acc = test(net, test_loader)
        print("=> Epoch 0 Test Acc: ", test_acc)

        """
        Passing only the fc layer to the optimizer. 
        This prevents lower layers from being effected by weight decay.
        """
        if args.resume_lp_ckpt.lower() != "none" and args.protocol in [
            "lp+ft",
            "vatlp+ft",
            "udplp+ft",
            "souplp+ft"
        ]:
            print()
            print("*=!" * 20)
            print("Loading Saved LP Ckpt")
            ckpt = torch.load(args.resume_lp_ckpt)
            if "soup" in args.protocol:
                strict =False
            else:
                strict=True 
            incomp, unexpected = net.load_state_dict(ckpt["state_dict"],strict=strict)
            print("Incompatible Keys: ", incomp)
            print("Unexpected Keys: ", unexpected)

            _, lp_train_acc = test(net, train_loader)
            _, lp_test_acc = test(net, test_loader)
            print("LP Train Acc: ", lp_train_acc)
            print("LP Test Acc: ", lp_test_acc)
            print("*=!" * 20)
            print()

        else:
            print("****************************")
            print("Commence Linear Probe Training!")
            print("****************************")
            print()
            if "vat" in args.protocol and args.resume_lp_ckpt.lower() == "none":
                print("=%" * 20)
                print("VAT LP TRAINING")
                print("=%" * 20)
                net = freeze_layers_for_lp(net)

                """
                Perform Linear Probe Training 
                """
                net, ckpt = linear_probe_vat(
                    args,
                    net,
                    train_loader,
                    test_loader,
                    args.train_aug,
                    train_transform=None,
                )

                """
                Save LP Final Ckpt.
                """
                s = "vatlp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(
                    save_name=save_name, epoch=args.epochs
                )
                save_path = os.path.join(args.save, s)
                torch.save(ckpt, save_path)

                lp_train_loss, lp_train_acc = test(net, train_loader)
                lp_test_loss, lp_test_acc = test(net, test_loader)
            
            elif "udp" in args.protocol and args.resume_lp_ckpt.lower() == "none":
                print("=%" * 20)
                print("UDP LP TRAINING")
                print("=%" * 20)
                net = freeze_layers_for_lp(net)

                """
                Perform Linear Probe Training 
                """
                net, ckpt = linear_probe_udp(
                    args,
                    net,
                    train_loader,
                    test_loader,
                    args.train_aug,
                    train_transform=None,
                )

                """
                Save LP Final Ckpt.
                """
                s = "udplp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(
                    save_name=save_name, epoch=args.epochs
                )
                save_path = os.path.join(args.save, s)
                torch.save(ckpt, save_path)

                _, lp_train_acc = test(net, train_loader)
                _, lp_test_acc = test(net, test_loader)

            elif "soup" in args.protocol and args.resume_lp_ckpt.lower() == "none":
                print("=#" * 20)
                print("SOUP LP TRAINING")
                print("=#" * 20)
                net = freeze_layers_for_lp(net)

                """
                Perform Linear Probe Training 
                """
                net, ckpt = linear_probe_soup(
                    args,
                    net,
                    train_loader,
                    test_loader,
                    args.train_aug,
                    train_transform=None,
                )

                """
                Save LP Final Ckpt.
                """
                s = "souplp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(
                    save_name=save_name, epoch=args.epochs
                )
                save_path = os.path.join(args.save, s)
                torch.save(ckpt, save_path)

                _, lp_train_acc = test(net, train_loader)
                _, lp_test_acc = test(net, test_loader)

            elif "vat" not in args.protocol and "soup" not in args.protocol and "udp" not in args.protocol and args.resume_lp_ckpt.lower() == "none":
                print("=*" * 60)
                print("STANDARD LP TRAINING")
                print("=*" * 60)

                net = freeze_layers_for_lp(net)

                net, ckpt = linear_probe_vanilla(
                    args,
                    net,
                    train_loader,
                    test_loader,
                    args.train_aug,
                    train_transform=None,
                ) 

                """
                Save LP Final Ckpt.
                """
                s = "lp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(
                    save_name=save_name, epoch=args.epochs
                )
                save_path = os.path.join(args.save, s)
                torch.save(ckpt, save_path)

                _, lp_train_acc = test(net, train_loader)
                _, lp_test_acc = test(net, test_loader)

    """
    Performing Fine-tuing Training!
    """
    if args.protocol in ["lp+ft", "ft", "lpfrz+ft", "vatlp+ft","udplp+ft","souplp+ft"]:
        if args.protocol == "lpfrz+ft":
            print("=> Freezing Classifier, Unfreezing All Other Layers!")
            net = unfreeze_layers_for_lpfrz_ft(net)
        else:
            print("=> Unfreezing All Layers")
            net = unfreeze_layers_for_ft(net)
        log_path = os.path.join(
            "{}/logs".format(PREFIX), "ft+" + save_name + "_training_log.csv"
        )
        """
        Select FT Augmentation Scheme.
        """
        if (
            args.protocol in ["lp+ft", "vatlp+ft","udplp+ft","souplp+ft"]
            and args.resume_lp_ckpt.lower() == "none"
        ):
            del (
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                train_transform,
                test_transform,
            )
        train_dataset = blendedCIFARMNIST(
            train=True,
            randomized=False,
            correlation_strength=args.correlation_strength,
            use_paired=True,
        )
        test_dataset = blendedCIFARMNIST(
            train=False,
            randomized=False,
            correlation_strength=args.correlation_strength,
            use_paired=True,
        )

        ft_train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        ft_test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        _, test_acc = test(net, ft_test_loader)
        print("Beginning FT!")
        print("=> Epoch 0 Test Acc: ", test_acc)

        optimizer = torch.optim.SGD(
            net.parameters(),
            args.ft_learning_rate,
            momentum=args.ft_momentum,
            weight_decay=args.ft_decay,
            nesterov=True,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.ft_epochs,
        )
        start_epoch = 0

        with open(log_path, "w") as f:
            f.write("epoch,time(s),train_loss,test_loss,test_error(%)\n")

        """
        Perform Fine-Tuning Training 
        """
        net, ckpt = train_loop(
            args=args,
            protocol="ft",
            save_name="ft+" + save_name,
            log_path=log_path,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            end_epoch=args.ft_epochs,
            train_loader=ft_train_loader,
            test_loader=ft_test_loader,
            train_aug=args.ft_train_aug,
            train_transform=None,
        )

        """
        Save FT Final Ckpt.
        """
        s = "ft+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(
            save_name=save_name, epoch=args.ft_epochs
        )
        save_path = os.path.join(args.save, s)
        torch.save(ckpt, save_path)
        _, ft_train_acc = test(net, ft_train_loader)
        _, ft_test_acc = test(net, ft_test_loader)
    """
    Perform ID + OOD Evaluation!
    """
    ood_dataset = blendedSTLMNIST(train=False, transform=None, use_paired=True)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    random_ood_dataset = blendedSTLMNIST(
        train=False, transform=None, randomized=True, use_paired=True
    )
    random_ood_loader = torch.utils.data.DataLoader(
        random_ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if ood_loader:
        _, ood_acc = test(net, ood_loader)
        _, rand_ood_acc = test(net, random_ood_loader)
    else:
        ood_acc = -1

    with open(
        "/usr/workspace/trivedi1/simplicity_experiments_aaai/logs/consolidated.csv", "a"
    ) as f:
        write_str = [
            save_name.replace("_", ","),
            args.eval_dataset,
            lp_train_acc,
            lp_test_acc,
            ft_train_acc,
            ft_test_acc,
            ood_acc,
            rand_ood_acc,
        ]
        write_str = [str(i) for i in write_str]
        write_str = ",".join(write_str)
        f.write("{}\n".format(write_str))
        print(write_str)


if __name__ == "__main__":
    main()

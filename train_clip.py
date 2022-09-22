from asyncio import protocols
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
from clip_model import ClipModel
from torch.utils.data.distributed import DistributedSampler
from vat_loss import VATLoss
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Wrapped into function so we 
can call it for fine-tuning too.
"""
PREFIX="/p/lustre1/trivedi1/compnets/classifier_playground/"
#PREFIX="/p/lustre1/trivedi1/compnets/classifier_playground/"
def train_loop(args,protocol,save_name,log_path, net, optimizer,scheduler,start_epoch,end_epoch,train_loader, test_loader, train_aug, train_transform):

    use_clip_mean = "clip" in args.arch
    best_acc = 0
    weight_dict_initial, _ = get_param_weights_counts(net, detach=True)
    #if 'ft' in protocol:
    ood_loader = get_oodloader(args=args,dataset = args.eval_dataset,use_clip_mean=use_clip_mean)
    print('=> Beginning training from epoch:', start_epoch + 1)
    l2sp_loss = -1 
    if args.l2sp_weight != -1:
        print("=> Using l2sp weight: ",args.l2sp_weight)
        l2sp_loss = 0 
    if train_aug in ['cutmix','mixup','cutout']:
        transform = train_transform
    else:
        transform = None
    if train_aug in ['cutmix','mixup']:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if protocol in ['lp'] or not args.train_batchnorm: #note: protocol re-specified in the main function as lp or ft ONLY. 
        print("****** Freezing Batchnorm Parameters ******") 
    else:
        print("****** Updating Batchnorm Parameters ****") 
    early_stopping_epoch = 0
    for epoch in range(start_epoch, end_epoch):
        begin_time = time.time() 
        if protocol in ['lp'] or not args.train_batchnorm: #note: protocol re-specified in the main function as lp or ft ONLY. 
            net.eval()
        else:
            net.train()
        loss_ema = 0.
        for _, (images, targets) in tqdm.tqdm(enumerate(train_loader),disable=True):
            optimizer.zero_grad()
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            #use cutmix or mixup
            if transform:
                if train_aug in ['cutmix','mixup']:
                    images, targets= transform(images,target=targets)
                if train_aug == 'cutout':
                    images = transform(images)
            logits = net(images)
            loss = criterion(logits, targets)
            if args.l2sp_weight != -1:
                weight_dict, _ = get_param_weights_counts(net, detach=False)
                l2sp_loss = args.l2sp_weight * get_l2_dist(weight_dict_initial, weight_dict)
                loss += l2sp_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1

        test_loss, test_acc = test(net, test_loader)
        # train_loss, train_acc = test(net, train_loader)
        
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if is_best and epoch > 10:
            early_stopping_epoch = epoch
            checkpoint = {
            'epoch': epoch,
            'dataset': args.dataset,
            'model': args.arch,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'protocol':args.protocol
            }
            save_path = os.path.join(args.save, save_name + "_" + args.protocol +'_model_best.pth.tar')
            torch.save(checkpoint, save_path)

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))
        # if 'ft' not in protocol:
        # print(
        #     'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        #     ' Test Error {4:.2f}'
        #     .format((epoch + 1), int(time.time() - begin_time), loss_ema,
        #             test_loss, 100 - 100. * test_acc))
        #Print the OOD acc each epoch if we are fine-tuning.
        # else:
        # _,ood_acc = test(net,ood_loader)
        ood_acc = -1
        # _,ood_acc = test(net,ood_loader)
        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | L2 Loss {4:.3f} |'
            ' Test Error {5:.2f} | OOD Error {6:.2f} | EarlyStop {7}'
            .format((epoch + 1), int(time.time() - begin_time), loss_ema,
                    test_loss, l2sp_loss, 100 - 100. * test_acc,100 - 100. * ood_acc,early_stopping_epoch))

    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'protocol':args.protocol
    }
    return net,checkpoint                              

def extract_features(args, model,loader,train_aug,train_transform):
    if train_aug in ['cutmix','mixup','cutout']:
        transform = train_transform
    else:
        transform = None
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm.tqdm(loader,disable=True):
            data = data.cuda()
            if transform:
                if train_aug in ['cutmix','mixup']:
                    images, targets= transform(images,target=targets)
                if train_aug == 'cutout':
                    images = transform(images)
            if args.arch == 'resnet50':
                #using a timm model. called the 
                reps = torch.nn.functional.adaptive_avg_pool2d(model.module.forward_features(data),1)
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
            if "clip" in args.arch:
                #using a timm model. called the
                reps = model.module.get_features(data)
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
    # features = torch.nn.functional.adaptive_avg_pool2d(np.squeeze(np.concatenate(features)),1)
    features = np.squeeze(np.concatenate(features))
    labels = np.concatenate(labels)
    return features, labels


def linear_probe_vat(args, net, train_loader,test_loader, train_aug, train_transform):
    net.eval()
    train_features, train_labels = extract_features(args,model=net, loader=train_loader,train_aug=train_aug,train_transform=train_transform)
    test_features, test_labels = extract_features(args,model=net, loader=test_loader,train_aug='test',train_transform=None)
    print(train_features.shape)

    rep_train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_features),torch.Tensor(train_labels).long())
    rep_train_dataloader = torch.utils.data.DataLoader(rep_train_dataset,batch_size=args.batch_size,shuffle=True) 
    
    rep_test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_features),torch.Tensor(test_labels).long())
    rep_test_dataloader = torch.utils.data.DataLoader(rep_test_dataset,batch_size=args.batch_size,shuffle=True) 
    
    """
    Create a linear probe layer.
    We will attach it separately back.
    """
    
    fc = torch.nn.Linear(train_features.shape[1],NUM_CLASSES_DICT[args.dataset]).to(DEVICE)
    optimizer = torch.optim.SGD(
        fc.parameters(),
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
        _,train_acc = test(fc, rep_train_dataloader) 
        _,val_acc = test(fc, rep_test_dataloader) 
        
        vat_avg /= len(rep_train_dataloader) 
        consistency_loss_avg /= len(rep_train_dataloader) 
        loss_avg /= len(rep_train_dataloader) 
        print("Epoch: {4} -- VAT: {0:.4f} -- Con: {1:.4f} -- Tot.:{2:.4f} -- Train Acc: {3:.4f} -- Test Acc: {5:.4f}".format(vat_avg,consistency_loss_avg,loss_avg,train_acc,epochs,val_acc))
    #Set the classifier weights
    net.module.fc = fc 
   
    """
    Compute acc. using forward passes.
    """
    _,acc = test(net=net,test_loader=test_loader)
    print("Completed VAT Training: {0:.3f}".format(acc))
    checkpoint = {
        'lp_epoch': args.epochs,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'best_acc': acc,
        'protocol':'vatlp'
    }
    return net, checkpoint

def linear_probe_fgsm(args, net, train_loader,test_loader, train_aug, train_transform):
    net.eval()
    train_features, train_labels = extract_features(args,model=net, loader=train_loader,train_aug=train_aug,train_transform=train_transform)
    test_features, test_labels = extract_features(args,model=net, loader=test_loader,train_aug='test',train_transform=None)
    print(train_features.shape)

    rep_train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_features),torch.Tensor(train_labels).long())
    rep_train_dataloader = torch.utils.data.DataLoader(rep_train_dataset,batch_size=args.batch_size,shuffle=True) 
    
    rep_test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_features),torch.Tensor(test_labels).long())
    rep_test_dataloader = torch.utils.data.DataLoader(rep_test_dataset,batch_size=args.batch_size,shuffle=True) 
    
    """
    Create a linear probe layer.
    We will attach it separately back.
    """
    
    fc = torch.nn.Linear(train_features.shape[1],NUM_CLASSES_DICT[args.dataset]).to(DEVICE)
    optimizer = torch.optim.SGD(
        fc.parameters(),
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
        for batch_idx, (data, target) in enumerate(rep_train_dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            adv_data = fast_gradient_method(fc, data, args.eps, np.inf)
            output = fc(adv_data)
            loss = criterion(output, target) 
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += loss 
        _,train_acc = test(fc, rep_train_dataloader) 
        _,test_acc = test(fc, rep_test_dataloader) 
        
        print("Epoch: {0} -- Loss: {1:.4f} -- Train Acc: {2:.4f} -- Test Acc: {3:.4f}".format(epochs,loss_avg,train_acc,test_acc))
    #Set the classifier weights
    net.module.fc = fc 
   
    """
    Compute acc. using forward passes.
    """
    _,acc = test(net=net,test_loader=test_loader)
    print("Completed FGSM Training: {0:.3f}".format(acc))
    checkpoint = {
        'lp_epoch': args.epochs,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'best_acc': acc,
        'protocol':'vatlp'
    }
    return net, checkpoint

def linear_probe_udp(args, net, train_loader, test_loader, train_aug, train_transform):

    net.eval()
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
                attack_iters=20, 
                rs=False, 
                clamp_val=None, 
                use_alpha_scheduler=False, 
                sample_iters='none')
            optimizer.zero_grad()
            clean_output = fc(data)
            adv_output = fc(data+delta)
            loss_clean = criterion(clean_output, target)
            loss_adv =  args.loss_weight * criterion(adv_output,target)
            #print(loss_clean, loss_adv) 
            loss = loss_clean + loss_adv 
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += loss
            loss_adv_avg += loss_adv
            loss_clean_avg += loss_clean 
        _, train_acc = test(fc, rep_train_dataloader)
        _, val_acc = test(fc, rep_test_dataloader)

        loss_avg /= len(rep_train_dataloader)
        loss_clean_avg /= len(rep_train_dataloader)
        loss_adv_avg /= len(rep_train_dataloader)
        print(
            "UDP, Epoch: {0} -- Loss: {1:.3f} -- Clean: {2:.3f} -- UDP: {3:.3f} -- Train Error: {4:.4f} -- Test Error: {5:.4f} -- ".format(
                epochs,
                loss_avg,                
                loss_clean_avg,
                loss_adv_avg,
                100 - 100.0 * train_acc,
                100 - 100.0 * val_acc,
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


def main():
    args = arg_parser()
    for arg in sorted(vars(args)):
        print("=> " ,arg, getattr(args, arg))
    print("=> Device: ",DEVICE)
    print("=> Num GPUS: ",torch.cuda.device_count())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)


    if args.pretrained_ckpt == "None":
        if args.arch.lower()  == 'resnet50':
            print("\t*** Using Default Pretrained Model!!")
            net = timm.create_model(args.arch,pretrained=True)
        if 'clip' in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ",encoder_type)
            print("\t => Using Scratch Clip Encoder!") 
            net = ClipModel(model_name=encoder_type,scratch=True)
    else:
        if args.arch.lower() == 'resnet50':
            net = timm.create_model(args.arch,pretrained=False)
            net = load_moco_ckpt(model=net, args=args)
            print("\t*** Using MoCoV2 RN50 Pretrained Model!!")
        if 'clip' in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ",encoder_type)
            print("\t => Using Default Clip Ckpt!!") 
            net = ClipModel(model_name=encoder_type,scratch=False)
    use_clip_mean = "clip" in args.arch 
    if "vat" in args.protocol:
        lp_aug_name = "vat-{}".format(args.alpha)
    elif "fgsm" in args.protocol:
        lp_aug_name = "fgsm-{}".format(args.eps)

    else:
        lp_aug_name = args.train_aug
    save_name =  args.dataset \
        + '_' + args.arch \
        + '_'+ args.protocol \
        + "_" + lp_aug_name \
        + "_" + args.ft_train_aug \
        + "_" + str(args.epochs) \
        + "_" + str(args.learning_rate) \
        + "_" + str(args.decay) \
        + "_" + str(args.ft_epochs) \
        + "_" + str(args.ft_learning_rate) \
        + "_" + str(args.ft_decay) \
        + "_" + str(args.l2sp_weight) \
        + "_" + str(args.seed) \
        + "_" + str(args.train_batchnorm)
    
    print("******************************")
    print(save_name)
    print("******************************")
 
    """
    Throw away classifier.
    Create new classifier with number of classes.
    """
    net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
    print('Reset Classifer: ',net.get_classifier()) 
    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    torch.backends.cudnn.benchmark = True

    """
    Performing Linear Probe Training!
    """
    lp_train_acc, lp_test_acc, lp_train_loss, lp_test_loss = -1,-1,-1,-1
    ft_train_acc, ft_test_acc, ft_train_loss, ft_test_loss = -1,-1,-1,-1
    if args.protocol in ['lp','lp+ft','vatlp+ft','vatlp','fgsmlp','fgsmlp+ft']:

        log_path = os.path.join("{}/logs".format(PREFIX),
                            "lp+" + save_name + '_training_log.csv')

        """
        Select Augmentation Scheme.
        """
        train_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.train_aug,use_clip_mean=use_clip_mean) 
        test_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.test_aug,use_clip_mean=use_clip_mean) 
        train_loader, test_loader = get_dataloaders(args=args, 
            train_aug=args.train_aug,
            test_aug=args.test_aug, 
            train_transform=train_transform,
            test_transform=test_transform,
            use_clip_mean=use_clip_mean) 
        NUM_CLASSES = NUM_CLASSES_DICT[args.dataset]    
        print("=> Num Classes: ",NUM_CLASSES) 
        print("=> Train: ",train_loader.dataset) 
        print("=> Test: ",test_loader.dataset) 

        test_loss, test_acc = test(net, test_loader)
        print("=> Epoch 0 Test Acc: ",test_acc)

        """
        Passing only the fc layer to the optimizer. 
        This prevents lower layers from being effected by weight decay.
        """
        if args.resume_lp_ckpt.lower() != "none" and args.protocol in ['lp+ft','vatlp+ft','fgsm+ft']:
            print("****************************")
            print("Loading Saved LP Ckpt")
            ckpt = torch.load(args.resume_lp_ckpt)
            incomp, unexpected = net.load_state_dict(ckpt['state_dict'])
            print("Incompatible Keys: ",incomp)
            print("Unexpected Keys: ",unexpected)

            lp_train_loss, lp_train_acc = test(net, train_loader)
            lp_test_loss, lp_test_acc = test(net, test_loader)
            print("LP Train Acc: ",lp_train_acc)
            print("LP Test Acc: ",lp_test_acc)
            print("****************************")

        else:
            print("****************************")
            print("Commence Linear Probe Training!")
            print("****************************")
            print()
            if "vat" in args.protocol and args.resume_lp_ckpt.lower() == "none" :
                print("=%"*20)
                print("VAT LP TRAINING")
                print("=%"*20)
                net = freeze_layers_for_lp(net)

                """
                Perform Linear Probe Training 
                """
                net, ckpt = linear_probe_vat(args, net, train_loader, test_loader, args.train_aug, train_transform)

                """
                Save LP Final Ckpt.
                """
                s = "vatlp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(save_name=save_name,epoch=args.epochs)
                save_path = os.path.join(args.save, s)
                torch.save(ckpt, save_path)

                lp_train_loss, lp_train_acc = test(net, train_loader)
                lp_test_loss, lp_test_acc = test(net, test_loader)
            elif "fgsm" in args.protocol and args.resume_lp_ckpt.lower() == "none" :
                print("=%"*20)
                print("FGSM LP TRAINING")
                print("=%"*20)
                net = freeze_layers_for_lp(net)

                """
                Perform Linear Probe Training 
                """
                net, ckpt = linear_probe_fgsm(args, net, train_loader, test_loader, args.train_aug, train_transform)

                """
                Save LP Final Ckpt.
                """
                s = "fgsmlp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(save_name=save_name,epoch=args.epochs)
                save_path = os.path.join(args.save, s)
                torch.save(ckpt, save_path)

                lp_train_loss, lp_train_acc = test(net, train_loader)
                lp_test_loss, lp_test_acc = test(net, test_loader)

            elif "vat" not in args.protocol and "fgsm" not in args.protocol and args.resume_lp_ckpt.lower() == "none" :
                print("=*"*60)
                print("STANDARD LP TRAINING")
                print("=*"*60)

                net = freeze_layers_for_lp(net)
                optimizer = torch.optim.SGD(
                    net.module.fc.parameters(),
                    args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.decay,
                    nesterov=True)

                start_epoch = 0
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                #     optimizer,
                #     T_max = args.epochs,
                #     eta_min = 1e-5,
                # )
            
                scheduler = LR_Scheduler(
                    optimizer,
                    warmup_epochs=0, warmup_lr = 0*args.batch_size/256, 
                    num_epochs=args.epochs, base_lr=args.learning_rate*args.batch_size/256, 
                    final_lr =1e-5 *args.batch_size/256, 
                    iter_per_epoch= len(train_loader),
                    constant_predictor_lr=False
                )
                with open(log_path, 'w') as f:
                    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

                """
                Perform Linear Probe Training 
                """
                net, ckpt = train_loop(args = args,
                    protocol='lp',
                    save_name = "lp+"+save_name,
                    log_path=log_path,
                    net = net, 
                    optimizer = optimizer,
                    scheduler = scheduler,
                    start_epoch = start_epoch,
                    end_epoch = args.epochs,
                    train_loader = train_loader, 
                    test_loader = test_loader, 
                    train_aug = args.train_aug, 
                    train_transform=train_transform)

                """
                Save LP Final Ckpt.
                """
                s = "lp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(save_name=save_name,epoch=args.epochs)
                save_path = os.path.join(args.save, s)
                torch.save(ckpt, save_path)

                lp_train_loss, lp_train_acc = test(net, train_loader)
                lp_test_loss, lp_test_acc = test(net, test_loader)
            else:
                print("ERROR ERROR ERROR INVALID LP PROTOCOL")
                print("EXITING")
    """
    Performing Fine-tuing Training!
    """
    if args.protocol in ['lp+ft','ft','lpfrz+ft','vatlp+ft','fgsmlp+ft']:
        if args.protocol == 'lpfrz+ft':
            print("=> Freezing Classifier, Unfreezing All Other Layers!")
            net = unfreeze_layers_for_lpfrz_ft(net)
        else: 
            print("=> Unfreezing All Layers") 
            net = unfreeze_layers_for_ft(net)
        log_path = os.path.join("{}/logs".format(PREFIX),
                            "ft+" + save_name + '_training_log.csv') 
        """
        Select FT Augmentation Scheme.
        """
        if args.protocol in ['lp+ft','vatlp+ft','fgsmlp+ft'] and args.resume_lp_ckpt.lower() == 'none':
            del train_loader, test_loader, optimizer, scheduler, train_transform, test_transform
        ft_train_transform = get_transform(dataset=args.dataset, 
            SELECTED_AUG=args.ft_train_aug,
            use_clip_mean=use_clip_mean)

        ft_test_transform = get_transform(dataset=args.dataset, 
            SELECTED_AUG=args.ft_test_aug,
            use_clip_mean=use_clip_mean)
             
        ft_train_loader, ft_test_loader = get_dataloaders(args=args, 
            train_aug=args.ft_train_aug,
            test_aug=args.ft_test_aug, 
            train_transform=ft_train_transform,
            test_transform=ft_test_transform,
            use_clip_mean=use_clip_mean) 

        test_loss, test_acc = test(net, ft_test_loader)
        print("=> Epoch 0 Test Acc: ",test_acc)
        
        optimizer = torch.optim.SGD(
            net.parameters(),
            args.ft_learning_rate,
            momentum=args.ft_momentum,
            weight_decay=args.ft_decay,
            nesterov=True)
         
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = args.ft_epochs,
        )
        start_epoch = 0

        with open(log_path, 'w') as f:
            f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

        """
        Perform Fine-Tuning Training 
        """ 
        net, ckpt = train_loop(args =args,
            protocol='ft',
            save_name = "ft+"+save_name,
            log_path=log_path,
            net = net, 
            optimizer = optimizer,
            scheduler = scheduler,
            start_epoch = start_epoch,
            end_epoch = args.ft_epochs,
            train_loader = ft_train_loader, 
            test_loader = ft_test_loader, 
            train_aug = args.ft_train_aug, 
            train_transform=ft_train_transform)

        """
        Save FT Final Ckpt.
        """
        s = "ft+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(save_name=save_name,epoch=args.ft_epochs)
        save_path = os.path.join(args.save, s)
        torch.save(ckpt, save_path) 
        ft_train_loss, ft_train_acc = test(net, ft_train_loader)
        ft_test_loss, ft_test_acc = test(net, ft_test_loader)
    """
    Perform ID + OOD Evaluation!
    """
    ood_loader = get_oodloader(args=args,dataset=args.eval_dataset,use_clip_mean=use_clip_mean)
    if ood_loader:
        ood_loss, ood_acc = test(net, ood_loader)
    else:
        ood_loss, ood_acc = -1, -1
        
    with open("logs/consolidated.csv","a") as f:
        write_str = [save_name.replace("_",","),
            args.eval_dataset,
            lp_train_acc,
            lp_test_acc,
            ft_train_acc,
            ft_test_acc,
            ood_acc,
            lp_train_loss,
            lp_test_loss,
            ft_train_loss,
            ft_test_loss,
            ood_loss]
        write_str = [str(i) for i in write_str]
        write_str = ",".join(write_str)
        f.write("{}\n".format(write_str))
        print(write_str)

if __name__ == '__main__':
    main()
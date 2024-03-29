from collections import OrderedDict
import copy
import os
import time

from sklearn.linear_model import LogisticRegression,SGDClassifier
from clip_model import ClipModel
from vat_loss import VATLoss
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
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PREFIX="/p/lustre1/trivedi1/compnets/classifier_playground/"
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
    for epoch in range(start_epoch, end_epoch):
        begin_time = time.time() 
        if protocol in ['lp']: #note: protocol re-specified in the main function as lp or ft ONLY. 
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
        
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

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
        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | L2 Loss {4:.3f} |'
            ' Test Error {5:.2f} | OOD Error {6:.2f}'
            .format((epoch + 1), int(time.time() - begin_time), loss_ema,
                    test_loss, l2sp_loss, 100 - 100. * test_acc,100 - 100. * ood_acc))

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

def load_dp_ckpt(model,pretrained_ckpt):
    ckpt = torch.load(pretrained_ckpt)
    new_keys = [(k, "module.{}".format(k)) for k in list(ckpt['state_dict'].keys())]
    for old_k, new_k in new_keys:
        ckpt['state_dict'][new_k] = ckpt['state_dict'].pop(old_k)
    incompatible, unexpected = model.load_state_dict(ckpt['state_dict'],strict=False)
    print("Incompatible Keys: ", incompatible)
    print("Unexpected Keys: ",unexpected)
    return model

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
        if args.arch  == 'ResNet50':
            print("\t*** Using Default Pretrained Model!!")
            net = timm.create_model(args.arch,pretrained=True)
        if 'clip' in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ",encoder_type)
            print("\t => Using Scratch Clip Encoder!") 
            net = ClipModel(model_name=encoder_type,scratch=True)
    else:
        if args.arch == 'ResNet50':
            net = timm.create_model(args.arch,pretrained=False)
            net = load_moco_ckpt(model=net, args=args)
        if 'clip' in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ",encoder_type)
            print("\t => Using Default Clip Ckpt!!") 
            net = ClipModel(model_name=encoder_type,scratch=False)
    use_clip_mean = "clip" in args.arch 
 
    save_name =  args.dataset \
        + '_' + args.arch \
        + '_'+ args.protocol \
        + "_" + args.train_aug \
        + "_" + args.ft_train_aug \
        + "_" + str(args.epochs) \
        + "_" + str(args.learning_rate) \
        + "_" + str(args.decay) \
        + "_" + str(args.ft_epochs) \
        + "_" + str(args.ft_learning_rate) \
        + "_" + str(args.ft_decay) \
        + "_" + str(args.l2sp_weight) \
        + "_" + str(args.seed) \
        + "_" + str(args.alpha) #term specific for VAT to adjust reguarlization weight
    
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
    net = net.cuda()
    torch.backends.cudnn.benchmark = True

    """
    Performing Linear Probe Training!
    """
    lp_train_acc, lp_test_acc, lp_train_loss, lp_test_loss = -1,-1,-1,-1
    ft_train_acc, ft_test_acc, ft_train_loss, ft_test_loss = -1,-1,-1,-1
    if args.protocol in ['vatlp','vatlp+ft']:

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

        """
        Passing only the fc layer to optimizer 
        This prevents lower layers from being effected by weight decay.
        """
        if args.resume_lp_ckpt.lower() != "none" and args.protocol == 'vatlp+ft':
            print("****************************")
            print("Loading Saved vatlp Ckpt")
            ckpt = torch.load(args.resume_lp_ckpt)
            try:
                incomp, unexpected = net.load_state_dict(ckpt['state_dict'])
                print("Incompatible Keys: ",incomp)
                print("Unexpected Keys: ",unexpected)
            except:
                net =load_dp_ckpt(net, args.resume_lp_ckpt)

            lp_train_loss, lp_train_acc = test(net, train_loader)
            lp_test_loss, lp_test_acc = test(net, test_loader)
            print("LP Train Acc: ",lp_train_acc)
            print("LP Test Acc: ",lp_test_acc)
            print("****************************")

        else:
            print("****************************")
            print("Commence Linear Probe Training!")
            print("****************************")
            print("=> Freezing Layers!")
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

    """
    Performing Fine-tuing Training!
    """
    if args.protocol in ['vatlp+ft','ft']:
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
        if args.protocol == 'vatlp+ft' and args.resume_lp_ckpt.lower() == 'none':
            del train_loader, test_loader, optimizer, scheduler, train_transform, test_transform
        ft_train_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.ft_train_aug,use_clip_mean=use_clip_mean) 
        ft_test_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.ft_test_aug,use_clip_mean=use_clip_mean) 
        ft_train_loader, ft_test_loader = get_dataloaders(args=args, 
            train_aug=args.ft_train_aug,
            test_aug=args.ft_test_aug, 
            train_transform=ft_train_transform,
            test_transform=ft_test_transform,
            use_ft=True,
            use_clip_mean=use_clip_mean) 

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
    ood_loss, ood_acc = test(net, ood_loader)

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
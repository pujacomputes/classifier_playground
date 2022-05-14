from collections import OrderedDict
import copy
import os
import time

from sklearn.linear_model import LogisticRegression,SGDClassifier
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
We are going to use iterative sklearn regression. 
So each epoch will warm-start from the last one. 
"""

def get_dataloaders(args,train_aug, test_aug, train_transform,test_transform,use_ft=False,return_valoader=False):
    if train_aug not in ['mixup','cutmix','cutout']:
        #initialize the augmentation directly in the dataset
        if args.dataset == 'cifar10':
            """ 
            We will create a validation set.
            """
            train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
                train=True,
                transform=train_transform)
            if return_valoader:
                perm = np.random.RandomState(seed=1).permutation(len(train_dataset)) 
                num_train = int(0.8 * len(train_dataset))
                val_dataset = torch.utils.data.Subset(train_dataset,np.arange(40000,50000))
                train_dataset = torch.utils.data.Subset(train_dataset,np.arange(0,40000))
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
            if return_valoader:
                perm = np.random.RandomState(seed=1).permutation(len(train_dataset)) 
                num_val = int(0.2 * len(train_dataset))
                num_train = int(0.8 * len(train_dataset))
                train_dataset = torch.utils.data.Subset(train_dataset,perm[:num_train])
                val_dataset = torch.utils.data.Subset(train_dataset,perm[num_train:])
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
    if return_valoader:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
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
    if return_valoader: 
        return train_loader, val_loader,test_loader
    else:
        return train_loader, test_loader

def get_acc(preds, labels):
    return np.mean(preds == labels)


def normalize_features(features, normalize_index):
    # normalize_index is the index to compute mean and std-dev
    # TODO: consider changing to axis=0
    mean = np.mean(features[normalize_index])
    stddev = np.std(features[normalize_index])
    normalized_features = []
    for i in range(len(features)):
        normalized_features.append((features[i] - mean) / stddev)
    return normalized_features


def inv_normalize_weights(weights, intercept, features, normalize_index):
    # mean = np.mean(features[normalize_index], axis=0)
    # stddev = np.std(features[normalize_index], axis=0)
    # new_weights = weights / stddev
    # new_intercept = intercept - np.matmul(weights, mean / stddev)
    # Other version
    mean = np.mean(features[normalize_index])
    stddev = np.std(features[normalize_index])
    new_weights = weights / stddev
    new_intercept = intercept - np.matmul(weights, mean / stddev * np.ones(weights.shape[1]))
    return new_weights, new_intercept


def test_log_reg_warm_starting(train_features, train_labels,val_features, val_labels, num_cs=100, start_c=-7, end_c=2, max_iter=200, random_state=0):
    L = len(train_features)
    # TODO: figure out what this should be based on initial results.
    Cs = np.logspace(start_c, end_c, num_cs)
    clf = LogisticRegression(random_state=random_state, warm_start=True, max_iter=max_iter)
    #.fit(features[m][train_index], labels[m][train_index])
    accs = []
    best_acc = -1.0
    best_clf, best_coef, best_intercept, best_i, best_c = None, None, None, None, None
    cur_accs = []
    for i, C in zip(range(len(Cs)), Cs):
        clf.C = C
        clf.fit(train_features, train_labels)
        #cur_preds = clf.predict(test_features)
        train_acc = clf.score(train_features,train_labels)
        val_acc = clf.score(val_features,val_labels)
        key = (i,C,np.round(train_acc,5),np.round(val_acc,5))
        cur_accs.append(key)
        if val_acc > best_acc:
            best_acc = val_acc
            best_clf = copy.deepcopy(clf)
            best_coef = copy.deepcopy(clf.coef_)
            best_intercept = copy.deepcopy(clf.intercept_)
            best_i = i
            best_c = C
        
        print(key, flush=True)
    return best_clf, best_coef, best_intercept, best_c, best_i

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
    # features = torch.nn.functional.adaptive_avg_pool2d(np.squeeze(np.concatenate(features)),1)
    features = np.squeeze(np.concatenate(features))
    labels = np.concatenate(labels)
    return features, labels

def linear_prob_sk(args, net, train_loader, val_loader, test_loader, train_aug, train_transform):
    net.eval()
    # val_features, val_labels = extract_features(args,model=net, loader=val_loader,train_aug='test',train_transform=None)
    test_features, test_labels = extract_features(args,model=net, loader=test_loader,train_aug='test',train_transform=None)
    print(test_features.shape)

    ### normalize all features
    # normalized_train_features = normalize_features(train_features, 0)
    #normalized_val_features = normalize_features(val_features, 0)
    normalized_test_features = normalize_features(test_features, 0)

    num_cs=10
    start_c=-7
    end_c=2 
    max_iter=2000
    random_state=0 
    Cs = np.logspace(start_c, end_c, num_cs)
    clf = LogisticRegression(random_state=random_state, warm_start=True, max_iter=max_iter)
    accs = []
    best_acc = -1.0
    best_clf, best_coef, best_intercept, best_i, best_c = None, None, None, None, None
    cur_accs = []
    for i, C in zip(range(len(Cs)), Cs):
        clf.C = C
        train_features, train_labels = extract_features(args,model=net, 
            loader=train_loader,
            train_aug=train_aug,
            train_transform=train_transform)
        normalized_train_features = normalize_features(train_features, 0)
        val_features, val_labels = extract_features(args,model=net, 
            loader=val_loader,
            train_aug=train_aug,
            train_transform=train_transform)
        normalized_train_features = normalize_features(train_features, 0)
        normalized_val_features = normalize_features(val_features, 0)
        clf.fit(normalized_train_features, train_labels)
        train_acc = clf.score(normalized_train_features,train_labels)
        
        val_acc = clf.score(normalized_val_features,val_labels) #TODO: Return to val.
        test_acc = clf.score(normalized_test_features,test_labels) #TODO: Return to val.
        key = (i,C,clf.n_iter_,np.round(train_acc,5),np.round(val_acc,5),np.round(test_acc,5))
        cur_accs.append(key)
        if val_acc > best_acc:
            best_acc = val_acc
            best_clf = copy.deepcopy(clf)
            best_coef = copy.deepcopy(clf.coef_)
            best_intercept = copy.deepcopy(clf.intercept_)
            best_i = i
            best_c = C
        
        print(key, flush=True)    

    print("Best C: {0} Best i: {1}".format(best_c,best_i)) 
    new_coef, new_intercept = inv_normalize_weights(best_coef, best_intercept, train_features,
                                                    normalize_index=0) 
                    
    """
    Set the classifier weights.s
    """
    if args.arch == 'resnet50':
        fc = net.get_classifer()
        net.fc = set_linear_layer(layer=fc,coef=new_coef,intercept=new_intercept)
    else:
        print("CLIP not implemented!")

    """
    Compute acc. using forward passes.
    """
    acc = test(net=net,test_loader=test_loader)
    print("Post LogReg: {0:.3}".format(acc))
    checkpoint = {
        'epoch': -1,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'best_acc': acc,
        'protocol':'lp'
    }
    return net, checkpoint

def linear_prob_sk_sgd(args, net, train_loader, val_loader, test_loader, train_aug, train_transform):
    net.eval()
    # val_features, val_labels = extract_features(args,model=net, loader=val_loader,train_aug='test',train_transform=None)
    test_features, test_labels = extract_features(args,model=net, loader=test_loader,train_aug='test',train_transform=None)
    print(test_features.shape)

    ### normalize all features
    # normalized_train_features = normalize_features(train_features, 0)
    #normalized_val_features = normalize_features(val_features, 0)
    normalized_test_features = normalize_features(test_features, 0)

    num_cs=5
    start_c=-7
    end_c=2 
    max_iter=1000
    random_state=0 
    Cs = np.logspace(start_c, end_c, num_cs)
    clf = SGDClassifier(random_state=random_state, warm_start=True, max_iter=max_iter,loss='log_loss',penalty='l1')
    accs = []
    best_acc = -1.0
    best_clf, best_coef, best_intercept, best_i, best_c = None, None, None, None, None
    cur_accs = []
    for i, C in zip(range(len(Cs)), Cs):

        clf.alpha = C
        train_features, train_labels = extract_features(args,model=net, 
            loader=train_loader,
            train_aug=train_aug,
            train_transform=train_transform)
        normalized_train_features = normalize_features(train_features, 0)
        clf.partial_fit(normalized_train_features, train_labels,classes=np.unique(train_labels))
        train_acc = clf.score(normalized_train_features,train_labels)
        
        val_acc = clf.score(normalized_test_features,test_labels) #TODO: Return to val.
        key = (i,C,clf.n_iter_,np.round(train_acc,5),np.round(val_acc,5))
        cur_accs.append(key)
        if val_acc > best_acc:
            best_acc = val_acc
            best_clf = copy.deepcopy(clf)
            best_coef = copy.deepcopy(clf.coef_)
            best_intercept = copy.deepcopy(clf.intercept_)
            best_i = i
            best_c = C
        
        print(key, flush=True)    

    print("Best C: {0} Best i: {1}".format(best_c,best_i)) 
    new_coef, new_intercept = inv_normalize_weights(best_coef, best_intercept, train_features,
                                                    normalize_index=0) 
    """
    Set the classifier weights.s
    """
    if args.arch == 'resnet50':
        fc = net.fc
        net.fc = set_linear_layer(layer=fc,coef=new_coef,intercept=new_intercept)
    else:
        print("CLIP not implemented!")

    """
    Compute acc. using forward passes.
    """
    acc = test(net=net,test_loader=test_loader)
    print("Post LogReg: {0:.3}".format(acc))
    checkpoint = {
        'epoch': -1,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'best_acc': acc,
        'protocol':'lp'
    }
    return net, checkpoint

def linear_probe_vat(args, net, train_loader,test_loader, train_aug, train_transform):
    net.eval()
    if train_aug != 'test':
        print("*"*50)
        print("CAUTION: Train Aug should be set to TEST for VAT!")
        print("*"*50)
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
    fc = torch.nn.Linear(train_features.shape[1],10).to(DEVICE)
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
    #Set the classifier weights.s
    if args.arch == 'resnet50':
        net.module.fc = fc 
    else:
        print("CLIP not implemented!")

    """
    Compute acc. using forward passes.
    """
    _,acc = test(net=net,test_loader=test_loader)
    print("Post VAT: {0:.3f}".format(acc))
    checkpoint = {
        'epoch': -1,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'best_acc': acc,
        'protocol':'vatlp'
    }
    return net, checkpoint

def train_loop(args,protocol,save_name,log_path, net, optimizer,scheduler,start_epoch,end_epoch,train_loader, test_loader, train_aug, train_transform):
    best_acc = 0
    weight_dict_initial, _ = get_param_weights_counts(net, detach=True)
    #if 'ft' in protocol:
    ood_loader = get_oodloader(args=args,dataset = args.eval_dataset)
    print('=> Beginning training from epoch:', start_epoch + 1)
    l2sp_loss = -1 
    if args.l2sp_weight != -1:
        print("=> Using l2sp weight: ",args.l2sp_weight)
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
        for _, (images, targets) in enumerate(train_loader):
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

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # elif args.save != './ckpts':
    #     raise Exception('%s exists' % args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    if args.pretrained_ckpt == "None":
        net = timm.create_model(args.arch,pretrained=True)
    else:
        net = timm.create_model(args.arch,pretrained=False)
        net = load_moco_ckpt(model=net, args=args)
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

        log_path = os.path.join("/p/lustre1/trivedi1/compnets/classifier_playground/logs",
                            "lp+" + save_name + '_training_log.csv')

        """
        Select Augmentation Scheme.
        """
        train_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.train_aug) 
        test_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.test_aug) 
        train_loader, test_loader = get_dataloaders(args=args, 
            train_aug=args.train_aug,
            test_aug=args.test_aug, 
            train_transform=train_transform,
            test_transform=test_transform,
            return_valoader=False) 
        NUM_CLASSES = NUM_CLASSES_DICT[args.dataset]    
        print("=> Num Classes: ",NUM_CLASSES) 
        print("=> Train: ",train_loader.dataset) 
        print("=> Test: ",test_loader.dataset) 

        """
        Passing only the fc layer. 
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
                #append module to eerything and try again.


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
            # net, ckpt = linear_prob_sk_sgd(args, net, train_loader, val_loader, test_loader, args.train_aug, train_transform)
            net, ckpt = linear_probe_vat(args, net, train_loader, test_loader, args.train_aug, train_transform)
            save_name = "vatlp+"+save_name

            """
            Save LP Final Ckpt.
            """
            s = "vat+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(save_name=save_name,epoch=args.epochs)
            save_path = os.path.join(args.save, s)
            torch.save(ckpt, save_path)

            lp_train_loss, lp_train_acc = test(net, train_loader)
            lp_test_loss, lp_test_acc = test(net, test_loader)

    """
    Performing Fine-tuing Training!
    """
    if args.protocol in ['vatlp+ft','ft','vatlp+ft']:
        if args.protocol == 'lpfrz+ft':
            print("=> Freezing Classifier, Unfreezing All Other Layers!")
            net = unfreeze_layers_for_lpfrz_ft(net)
        else: 
            print("=> Unfreezing All Layers") 
            net = unfreeze_layers_for_ft(net)
        log_path = os.path.join("/p/lustre1/trivedi1/compnets/classifier_playground/logs",
                            "ft+" + save_name + '_training_log.csv') 
        """
        Select FT Augmentation Scheme.
        """
        if args.protocol == 'vatlp+ft' and args.resume_lp_ckpt.lower() == 'none':
            del train_loader, test_loader, optimizer, scheduler, train_transform, test_transform
        ft_train_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.ft_train_aug) 
        ft_test_transform = get_transform(dataset=args.dataset, SELECTED_AUG=args.ft_test_aug) 
        ft_train_loader, ft_test_loader = get_dataloaders(args=args, 
            train_aug=args.ft_train_aug,
            test_aug=args.ft_test_aug, 
            train_transform=ft_train_transform,
            test_transform=ft_test_transform,use_ft=True) 

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
    ood_loader = get_oodloader(args=args,dataset=args.eval_dataset)
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
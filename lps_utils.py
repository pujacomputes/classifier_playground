import time
import torch
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from vat_loss import VATLoss
from utils import *
import math 
import pdb
import os

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
        ood_acc = -1
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

def create_sparse_clfs(classifier_pool):
    num_dims = classifier_pool[0].weight.data.shape[1]
    sparsity_feats = num_dims // len(classifier_pool) 
    num_classes = classifier_pool[0].weight.data.shape[0]
    for enum, cls in enumerate(classifier_pool):
        """
        if we are at the end of the classifiers, 
        then we will just let the classifier use a few extra features.
        """

        start_idx = enum * sparsity_feats

        if enum == len(classifier_pool)-1:
            end_idx = cls.weight.data.shape[1]

        else:
            end_idx = (enum+1) * sparsity_feats

        """
        Now, set all weights to 0, and then replace specified indices, with these values. 
        """

        with torch.no_grad():
            empty = torch.zeros(num_classes,end_idx -start_idx)
            torch.nn.init.kaiming_uniform_(empty, a=math.sqrt(5))
            torch.nn.init.constant_(cls.weight.data,0.0)
            cls.weight[:,start_idx:end_idx] = empty
            assert empty.norm(), cls.weight.norm()
            # print(enum, start_idx, end_idx,np.round(cls.weight.norm().cpu().numpy(),4))
    return classifier_pool

def test_soup_reps(net, test_loader,args):
    """Evaluate network on given dataset."""
    net.eval()

    with torch.no_grad():
        if isinstance(net, torch.nn.DataParallel):
            sum_weight = torch.zeros_like(net.module.classifier_pool[0].weight.data)
            if args.use_bias: 
                bias_weight = torch.zeros_like(net.module.classifier_pool[0].bias.data)
            num_clfs = len(net.module.classifier_pool) 
        else:
            sum_weight = torch.zeros_like(net.classifier_pool[0].weight.data)
            bias_weight = torch.zeros_like(net.classifier_pool[0].bias.data)
            num_clfs = len(net.classifier_pool) 

        if isinstance(net, torch.nn.DataParallel):
            for cls in net.module.classifier_pool:
                sum_weight += cls.weight.data

                if args.use_bias: 
                    bias_weight += cls.bias.data
        else:
            for cls in net.classifier_pool:
                sum_weight += cls.weight.data
                bias_weight += cls.bias.data
        sum_weight = torch.div(sum_weight, num_clfs)
        if args.use_bias: 
            bias_weight = torch.div(bias_weight, num_clfs)

        avg_cls = torch.nn.Linear(in_features=net.module.classifier_pool[0].in_features,out_features=net.module.classifier_pool[1].out_features,bias=args.use_bias)
    
        if args.use_bias:
            avg_cls.weight = torch.nn.Parameter(sum_weight,requires_grad = False)
            avg_cls.bias = torch.nn.Parameter(bias_weight, requires_grad = False)
        else:
            avg_cls.weight = torch.nn.Parameter(sum_weight,requires_grad = False)
        net.avg_cls = avg_cls.cuda()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda() 
                   
            logits = net.avg_cls(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
    return np.round(total_loss / len(test_loader),5), np.round(total_correct / len(test_loader.dataset),5)

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
    if isinstance(net, torch.nn.DataParallel):
        net.module.classifier_pool = classifier_pool

        optimizer = torch.optim.SGD(
            net.module.classifier_pool.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.decay,
            nesterov=True)
    else: 
        net.classifier_pool = classifier_pool
        optimizer = torch.optim.SGD(
            net.classifier_pool.parameters(),
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
            if isinstance(net, torch.nn.DataParallel):
                for cls in net.module.classifier_pool:
                    loss += avg_val * criterion(cls(data),target) 
            else: 
                for cls in net.classifier_pool:
                    loss += avg_val * criterion(cls(data),target)
             
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += loss 
        _,train_acc = test_soup_reps(net, rep_train_dataloader,args) 
        _,test_acc = test_soup_reps(net, rep_test_dataloader,args) 
        
        print("Epoch: {0} -- Loss: {1:.4f} -- Train Acc: {2:.4f} -- Test Acc: {3:.4f}".format(epochs,loss_avg,train_acc,test_acc))
    """
    After training is complete, we are going to set the fully connected layer.
    Right now, I am taking the average soup. But, I'll save the classifier pool
    in the checkpoint so that other soup recipes can be applied.
    """  
    #make average soup.
    print("=> Making Avg. Soup")
    net.eval()
    with torch.no_grad():
        if isinstance(net, torch.nn.DataParallel):
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
        else:
            sum_weight = torch.zeros_like(net.classifier_pool[0].weight.data)
            bias_weight = torch.zeros_like(net.classifier_pool[0].bias.data)

            for cls in net.classifier_pool:
                sum_weight += cls.weight.data.detach().clone()
                bias_weight += cls.bias.data.detach().clone()
            sum_weight = torch.div(sum_weight, len(net.classifier_pool))
            bias_weight= torch.div(bias_weight, len(net.classifier_pool))

        avg_cls = torch.nn.Linear(in_features=sum_weight.shape[0],out_features=sum_weight.shape[1],bias=args.use_bias)
        avg_cls.weight = torch.nn.Parameter(sum_weight)
        if args.use_bias:
            avg_cls.bias = torch.nn.Parameter(bias_weight)
        avg_cls.requires_grad = False  
    """
    Make Greedy Soup
    """
    print("=> Making Greedy Soup")
    acc_list = []
    for classifier in net.module.classifier_pool:
        if isinstance(net, torch.nn.DataParallel):
            net.module.fc = classifier
        else:
            net.fc = classifier
        _, acc = test(net=net,test_loader=test_loader)
        acc_list.append(np.round(acc,4))
    print("Accs: ",acc_list)
    idx = np.argsort(-1.0 * np.array(acc_list))
    if isinstance(net, torch.nn.DataParallel):
        greedy_cls = torch.nn.Linear(net.module.classifier_pool[idx[0]].weight.shape[0],classifier.weight.shape[1],bias=args.use_bias)
        greedy_cls.weight =  torch.nn.Parameter(net.module.classifier_pool[idx[0]].weight.data.detach().clone(),requires_grad=False) 
        if args.use_bias:
            greedy_cls.bias =  torch.nn.Parameter(net.module.classifier_pool[idx[0]].bias.data.detach().clone(),requires_grad=False) 
        
        temp_cls = torch.nn.Linear(net.module.classifier_pool[idx[0]].weight.shape[0],classifier.weight.shape[1],bias=args.use_bias)
        temp_cls.weight =  torch.nn.Parameter(net.module.classifier_pool[idx[0]].weight.data.detach().clone(),requires_grad=False) 
        if args.use_bias:
            temp_cls.bias=  torch.nn.Parameter(net.module.classifier_pool[idx[0]].bias.data.detach().clone(),requires_grad=False) 
    else:
        greedy_cls = torch.nn.Linear(net.classifier_pool[idx[0]].weight.shape[0],classifier.weight.shape[1],bias=args.use_bias)
        greedy_cls.weight =  torch.nn.Parameter(net.classifier_pool[idx[0]].weight.data.detach().clone(),requires_grad=False) 
        if args.use_bias:
            greedy_cls.bias =  torch.nn.Parameter(net.classifier_pool[idx[0]].bias.data.detach().clone(),requires_grad=False) 
        
        temp_cls = torch.nn.Linear(net.classifier_pool[idx[0]].weight.shape[0],classifier.weight.shape[1],bias=args.use_bias)
        temp_cls.weight =  torch.nn.Parameter(net.classifier_pool[idx[0]].weight.data.detach().clone(),requires_grad=False) 
        if args.use_bias:
            temp_cls.bias=  torch.nn.Parameter(net.classifier_pool[idx[0]].bias.data.detach().clone(),requires_grad=False) 
    curr_acc = acc_list[idx[0]]
    for enum,i in enumerate(idx[1:]): 
        if isinstance(net, torch.nn.DataParallel):
            cls = net.module.classifier_pool[i]
        else:
            cls = net.classifier_pool[i]
        """
        We are going to combine classifier_pool!
        """
        temp_cls.weight = torch.nn.Parameter(0.5 * (greedy_cls.weight.detach().clone() + cls.weight.detach().clone()),requires_grad=False)
        if args.use_bias:
            temp_cls.bias = torch.nn.Parameter(0.5 * (greedy_cls.bias.detach().clone() + cls.bias.detach().clone()),requires_grad=False)
        if isinstance(net, torch.nn.DataParallel):
            net.module.fc = temp_cls.cuda()
        else:
            net.fc = temp_cls
        _, acc = test(net=net,
            test_loader=test_loader)
        if acc > curr_acc:
            print("Adding {0} to soup -- {1:.3f}!".format(enum,curr_acc))
            curr_acc = acc
            greedy_cls.weight = torch.nn.Parameter(temp_cls.weight.clone().detach(),requires_grad=False)
            if args.use_bias:
                greedy_cls.bias = torch.nn.Parameter(temp_cls.bias.clone().detach(),requires_grad=False)
    net.soup_classifier = greedy_cls 

    if isinstance(net, torch.nn.DataParallel):
        net.module.fc = greedy_cls.cuda() 
    else:
        net.fc = greedy_cls
    _,soup_acc = test(net=net,test_loader=test_loader) 
    print('=> Greedy Soup Acc: ',soup_acc)

    """
    Compute acc. using forward passes.
    """
    if isinstance(net, torch.nn.DataParallel):
        net.module.fc = avg_cls.cuda()
    else: 
        net.fc = avg_cls 

    #net.module.fc = avg_cls 
    _,acc = test(net=net,test_loader=test_loader)
 
    print("=> Avg Soup Acc. : {0:.4f}".format(acc))
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
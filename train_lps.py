from asyncio import protocols
import os
import time
import torch
import timm
import tqdm 
import numpy as np
from utils import *
from clip_model import ClipModel
from lps_utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Wrapped into function so we
can call it for fine-tuning too.
"""
PREFIX="/p/lustre1/trivedi1/compnets/classifier_playground/"


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

    """
    Init/Load Model
    """

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
    elif "soup" in args.protocol:
        #TODO: this will be adjusted when other soup recipes are added!
        lp_aug_name = "soup-{}-{}".format(args.num_cls,args.use_bias)
    else:
        lp_aug_name = args.train_aug
    print("=> USE BIAS?: ",args.use_bias)
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
    Create new classifier with correct number of classes.
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
    if args.protocol in ['lp','lp+ft','vatlp+ft','vatlp','fgsmlp','fgsmlp+ft',"soup-avg-lp", 'soup-avg-lp+ft']:

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
        if args.resume_lp_ckpt.lower() != "none" and args.protocol in ['lp+ft','vatlp+ft','fgsm+ft','soup-avg-lp+ft']:
            print("****************************")
            print("Loading Saved LP Ckpt")
            ckpt = torch.load(args.resume_lp_ckpt)
            if "soup" in args.protocol:
                strict=False
            else:
                strict=True
            incomp, unexpected = net.load_state_dict(ckpt['state_dict'],strict=strict)
            if "soup" in args.protocol:
                net.module.classifier_pool = ckpt['classifier_pool']
            print("Incompatible Keys: ",incomp)
            print("Unexpected Keys: ",unexpected)

            if args.protocol == 'soup-avg-lp+ft':
                """
                Create the soup-classifier
                """
                print()
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
                    avg_cls = torch.nn.Linear(in_features=sum_weight.shape[0],out_features=sum_weight.shape[1],bias=args.use_bias)
                    avg_cls.weight = torch.nn.Parameter(sum_weight)
                    if args.use_bias:
                        avg_cls.bias = torch.nn.Parameter(bias_weight)
                    avg_cls.requires_grad = False                
                    net.module.fc = avg_cls
                    print("=> Avg. Cls has been set!")


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
            
            elif "soup" in args.protocol and args.resume_lp_ckpt.lower() == "none" :
                print("=%"*20)
                print("SOUP! LP TRAINING")
                print("=%"*20)
                net = freeze_layers_for_lp(net)

                """
                Perform Linear Probe Training 
                """
                net, ckpt = linear_probe_soup(args, net, train_loader, test_loader, args.train_aug, train_transform)

                """
                Save LP Final Ckpt.
                """
                s = "souplp+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(save_name=save_name,epoch=args.epochs)
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
    if args.protocol in ['lp+ft','ft','lpfrz+ft','vatlp+ft','fgsmlp+ft','soup-avg-lp+ft']:
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
        if args.protocol in ['lp+ft','vatlp+ft','fgsmlp+ft','soup-avg-lp+ft'] and args.resume_lp_ckpt.lower() == 'none':
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
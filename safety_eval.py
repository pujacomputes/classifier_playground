"""
1. CIFAR 100-C
2. CIFAR 100-P
3. Adversarial Attacks
4. Anamoly Detection 
"""
import os
import random
from re import I
import time
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from safety_utils import *
from tqdm import tqdm
from calibration_tools import *
import torchvision.datasets as dset
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from dtd_dataset import DTD

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
    return total_loss / len(test_loader), total_correct / len(test_loader.dataset)


def test_c(net, test_data, args):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  corrs = CBAR_CORRUPTIONS if 'Bar' in args.corruption_path else CORRUPTIONS
  for corruption in corrs:
    # Reference to original data is mutated
    # since we are modifying the dataset, I believe data will be normalized.
    test_data.data = np.load(args.corruption_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(args.corruption_path+ 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)

def test_cal(net, test_data, args):
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.to('cpu').numpy()

    corruption_accs = []
    calib_scores = []
    corrs = CBAR_CORRUPTIONS if 'Bar' in args.corruption_path else CORRUPTIONS
    for corruption in corrs:
        confidence = []
        correct = []
        # Reference to original data is mutated
        test_data.data = np.load(args.corruption_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(args.corruption_path+ 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        num_correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()

                output = net(data)

                # accuracy
                pred = output.data.max(1)[1]
                num_correct += pred.eq(target.data).sum().item()

                confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
                pred = output.data.max(1)[1]
                correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

        # acc, confidence, correct
        acc = num_correct / len(test_loader.dataset)
        confidence = np.array(confidence.copy())
        correct =  np.array(correct.copy())
        calib = calib_err(confidence=confidence, correct=correct, p='2')
        calib_scores.append(calib)
        corruption_accs.append(acc)
        print(corruption,np.round(calib,4),np.round(acc,4))
    acc = np.mean(corruption_accs)
    calib_score = np.mean(calib_scores)
    return acc, calib_score

def test_ood(net,test_loader, args):
    ood_num_examples = len(test_loader.dataset) // 5
    expected_ap = ood_num_examples / (ood_num_examples + len(test_loader.dataset))
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    auroc_list, aupr_list, fpr_list = [], [], []
    
    # /////////////// In Score /////////////// 
    in_score, right_score, wrong_score = get_ood_scores(net=net, 
        loader=test_loader, 
        ood_num_examples=ood_num_examples,
        args=args,
        in_dist=True)
    
    # /////////////// Gaussian Noise ///////////////

    dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    ood_data = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_batch_size, shuffle=True,
                                            num_workers=args.prefetch, pin_memory=True)

    print('\n\nGaussian Noise (sigma = 0.5) Detection')
    auroc, aupr, fpr = get_and_print_results(net=net,ood_loader = ood_loader,
        ood_num_examples=ood_num_examples,
        in_score=in_score, 
        args=args)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
    # /////////////// Rademacher Noise ///////////////

    dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    ood_data = torch.from_numpy(np.random.binomial(
        n=1, p=0.5, size=(ood_num_examples * args.num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_batch_size, shuffle=True)

    print('\n\nRademacher Noise Detection')
    auroc, aupr, fpr = get_and_print_results(net=net,ood_loader = ood_loader,
        ood_num_examples=ood_num_examples,
        in_score=in_score, 
        args=args)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    # /////////////// Blob ///////////////

    ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * args.num_to_avg, 32, 32, 3)))
    for i in range(ood_num_examples * args.num_to_avg):
        ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
        ood_data[i][ood_data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
    ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_batch_size, shuffle=True,
                                            num_workers=args.prefetch, pin_memory=True)

    print('\n\nBlob Detection')
    auroc, aupr, fpr = get_and_print_results(net=net,ood_loader = ood_loader,
        ood_num_examples=ood_num_examples,
        in_score=in_score, 
        args=args)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    # /////////////// Textures ///////////////
    # Textures should be part of TorchVision in the next release.
    ood_data = DTD(root="/p/lustre1/trivedi1/vision_data",
                                transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                                    transforms.ToTensor(), normalize]),
                                split='test',download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_batch_size, shuffle=True,
                                            num_workers=args.prefetch, pin_memory=True)

    print('\n\nTexture Detection')
    auroc, aupr, fpr = get_and_print_results(net=net,ood_loader = ood_loader,
        ood_num_examples=ood_num_examples,
        in_score=in_score, 
        args=args)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    # /////////////// SVHN ///////////////

    ood_data = torchvision.datasets.SVHN(root = "/p/lustre1/trivedi1/vision_data/SVHN",
        split='test',
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize]),
        download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_batch_size, shuffle=True,
                                            num_workers=args.prefetch, pin_memory=True)

    print('\n\nSVHN Detection')
    auroc, aupr, fpr = get_and_print_results(net=net,ood_loader = ood_loader,
        ood_num_examples=ood_num_examples,
        in_score=in_score, 
        args=args)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    # /////////////// Places69 ///////////////

    #use an ImageFolder.
    ood_data = torchvision.datasets.ImageFolder(root='/p/lustre1/trivedi1/vision_data/Places69',
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_batch_size, shuffle=True,
                                            num_workers=args.prefetch, pin_memory=True)

    print('\n\nPlaces69 Detection')
    auroc, aupr, fpr = get_and_print_results(net=net,ood_loader = ood_loader,
        ood_num_examples=ood_num_examples,
        in_score=in_score, 
        args=args)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    # /////////////// LSUN ///////////////
    ood_data = torchvision.datasets.LSUN(root="/p/lustre1/trivedi1/compnets/lsun/",
        classes='test',
        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                                    transforms.ToTensor(), normalize])
        )
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_batch_size, shuffle=True,
                                            num_workers=args.prefetch, pin_memory=True)

    print('\n\nLSUN Detection')
    auroc, aupr, fpr = get_and_print_results(net=net,ood_loader = ood_loader,
        ood_num_examples=ood_num_examples,
        in_score=in_score, 
        args=args)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    avg_auroc = np.mean(auroc_list)
    avg_aupr = np.mean(aupr_list)
    avg_fpr = np.mean(fpr_list)
    print('\n\nAverage OOD Detection')
    print_measures_ood(auroc = avg_auroc, 
        aupr = avg_aupr, 
        fpr = avg_fpr, 
        method_name=args.save_name, 
        recall_level=0.95)
    return avg_auroc,avg_aupr,avg_fpr

def test_p(net,args):
    if args.dataset == 'cifar100':
        num_classes=100
    elif args.dataset == 'cifar10':
        num_classes=10
    else:
        print("***** ERROR ERROR ERROR ******")
        print("Invalid Dataset Selected, Exiting")
        exit() 

    flip_list = []
    for p in ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
            'spatter', 'brightness', 'translate', 'rotate', 'tilt', 'scale']:
        
        dataset = torch.from_numpy(np.float32(np.load(os.path.join(args.perturb_path, p + '.npy')).transpose((0,1,4,2,3))))/255.
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=25, shuffle=False, num_workers=2, pin_memory=True)
        predictions = []

        with torch.no_grad():
            for data in loader:
                num_vids = data.size(0)
                data = data.view(-1,3,32,32).cuda()
                output = net(data * 2 - 1)

                for vid in output.view(num_vids, -1, num_classes):
                    predictions.append(vid.argmax(1).to('cpu').numpy())

            #TODO: adjust difficulty if necessary
            current_flip = flip_prob(predictions, difficulty=1, noise_perturbation=True if 'noise' in p else False)
            flip_list.append(current_flip)

            print('\n' + p, 'Flipping Prob',np.round(current_flip,4))

    print(flip_list)
    return np.mean(flip_list)

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

def arg_parser_eval():
    parser = argparse.ArgumentParser(
    description='Evaluates a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100'],
        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument(
        '--corruption_path',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/CIFAR-10-C/',
        required=False,
        help='Path to CIFAR and CIFAR-C directories')
    parser.add_argument(
        '--perturb_path',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/CIFAR-10-P/',
        required=False,
        help='Path to CIFAR and CIFAR-C directories')
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet50',
        choices=['resnet50','wrn', 'densenet', 'resnext'],
        help='Choose architecture.')
    parser.add_argument(
        '--ckpt',
        default = "/p/lustre1/trivedi1/compnets/classifier_playground/ckpts/ft+cifar10_resnet50_lp+ft_test_cutout_200_30.0_0.001_20_1e-05_0.0_-1_0_final_checkpoint_020_pth.tar",
        type=str,
        help='Checkpoint path for resume / test.')
    parser.add_argument('--eval_all', action='store_true', help='Performs all ML Safety evaluation.')
    parser.add_argument('--eval_C', action='store_true', help='Eval on Corruptions.')
    parser.add_argument('--eval_P', action='store_true', help='Eval on Perturbation.')
    parser.add_argument('--eval_A', action='store_true', help='Eval on Adversarial Examples.')
    parser.add_argument('--eval_O',action='store_true',help='Eval on OOD Datasets')
    parser.add_argument('--eval_Cal',action='store_true',help='Eval Calibration Error')
    parser.add_argument('--eval_Clean',action='store_true',help='Clean Accuracy')
    parser.add_argument('--use_xent',action='store_true',help='Use CrossEntropy in the OOD Eval.')
    parser.add_argument("--save_name",default='',help='Provide an identifier for the checkpoint')
    # data loader args
    parser.add_argument(
        '--eval_batch_size', default=128, type=int, help='Eval Batchsize')
    parser.add_argument(
        '--num_workers',default=8, type=int, help='Num Workers')
    parser.add_argument(
        '--num_to_avg',default=1, type=int, help='Num to Avg.')
    parser.add_argument(
        '--prefetch',action='store_true', help='Prefetch Ood Loader')
    args = parser.parse_args()
    return args 


def main():
    """
    Load the saved checkpoint.
    Load clean CIFAR. 

    Provide options to perform the other ML safety evaluations.
    We will need to use their checkpoints.
    """ 
    args = arg_parser_eval()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_acc, test_acc,ood_test_acc = -1,-1,-1

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
    test_transform = transforms.Compose(
        [transforms.Resize(224),transforms.ToTensor(), normalize])
    
    NUM_CLASSES = 10
    clean_test_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
    train=False,
    transform=test_transform) 
    
    clean_train_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",
    train=True,
    transform=test_transform) 
    
    net = timm.create_model(args.arch,pretrained=False)
    net.reset_classifier(10)
    if "moco" in args.ckpt:
        #Can't use data-parallel for feature extractor 
        net = load_moco_ckpt(model=net, pretrained_ckpt=args.ckpt).cuda()
    else:
        net = load_dp_ckpt(net,pretrained_ckpt=args.ckpt).cuda() #do this so there is not a conflict. 
    net_name = args.ckpt 

    print("=> Num GPUS: ", torch.cuda.device_count()) 
        # Fix dataloader worker issue
    # https://github.com/pytorch/pytorch/issues/5059
    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    clean_test_loader = torch.utils.data.DataLoader(
        clean_test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    

    safety_logs_prefix = "/p/lustre1/trivedi1/compnets/classifier_playground/safety_logs/"
    save_name = args.save_name 
    print("=> Save Name: ",save_name)
    """
    Clean Acc.
    """
    if args.eval_all or args.eval_Clean:
        clean_train_loader = torch.utils.data.DataLoader(
            clean_train_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)
    
        clean_err, clean_acc = test(net=net,test_loader=clean_train_loader,adv=None)
        print("=> Clean Train Acc: {0:.4f}".format(clean_acc))
        with open("{}/clean_train_acc.csv".format(safety_logs_prefix),"a") as f:
            write_str = "{save_name},{acc:.4f}\n".format(save_name = save_name,acc = clean_acc)
            f.write(write_str)
        del clean_train_loader
        del clean_train_dataset
        clean_test_err, clean_test_acc = test(net=net,test_loader=clean_test_loader,adv=None)
        print("=> Clean Test Acc: {0:.4f}".format(clean_test_acc))
        with open("{}/clean_test_acc.csv".format(safety_logs_prefix),"a") as f:
            write_str = "{save_name},{acc:.4f}\n".format(save_name = save_name,acc = clean_test_acc)
            f.write(write_str)
    """
    Anamoly Detection.
    """
    if args.eval_all or args.eval_O:
        auroc, aupr, fpr = test_ood(net,clean_test_loader, args)
        print("=> Anamoly AUROC: {0:.4f} -- AUPR: {1:.4f} -- FPR: {2:.4f}".format(auroc,aupr,fpr))        
        with open("{}/anamoly_detection.csv".format(safety_logs_prefix),"a") as f:
            write_str = "{save_name},{auroc:.4f},{aupr:.4f},{fpr:.4f}\n".format(save_name = save_name,auroc= auroc,aupr= aupr, fpr= fpr)
            f.write(write_str)

    """
    Adversarial Accuracy
    """
    if args.eval_all or args.eval_A:
        adversary = PGD(epsilon=2./255, num_steps=20, step_size=0.5/255).cuda()
        adv_test_loss, adv_test_acc = test(net, clean_test_loader, adv=adversary)
        print("=> Adv. Test Loss: {0:.4f} -- Adv. Test Acc: {1:.4f}".format(adv_test_loss,adv_test_acc))        
        with open("{}/adversaries.csv".format(safety_logs_prefix),"a") as f:
            write_str = "{save_name},{adv_test:.4f}\n".format(save_name=save_name,adv_test = adv_test_acc)
            f.write(write_str)

    """
    Corruptions Accuracy
    """
    if args.eval_all or args.eval_C:
        test_c_acc = test_c(net, clean_test_dataset, args)
        print("=> Corruption Test Acc: {0:.4f}".format(test_c_acc))
        with open("{}/corruptions.csv".format(safety_logs_prefix),"a") as f:
            write_str = "{save_name},{acc:.4f}\n".format(save_name=save_name,acc=test_c_acc)
            f.write(write_str)

    """
    Calibration Error.
    """
    if args.eval_all or args.eval_Cal:
        # Computed on CIFAR-100-C
        _, calib_err = test_cal(net, clean_test_dataset, args)
        print("=> Calib. Err.: {0:.4f}".format(calib_err))
        with open("{}/calibration.csv".format(safety_logs_prefix),"a") as f:
            write_str = "{save_name},{calib:.4f}\n".format(save_name=save_name,calib=calib_err)
            f.write(write_str)
    
    """
    Consistency Error.
    """
    if args.eval_all or args.eval_P:
        acc_P = test_p(net,args)
        print("=> Mean Flip Prob: {0:.4f}".format(acc_P)) 
        with open("{}/consistency.csv".format(safety_logs_prefix),"a") as f:
            write_str = "{save_name},{consist:.4f}\n".format(save_name=save_name,consist=acc_P)
            f.write(write_str)


if __name__ == '__main__':
    main()
import pdb
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import timm
import numpy as np
import argparse
import torch.nn as nn
import sklearn.metrics as sk
if torch.cuda.is_available():
    DEVICE = 'cuda'
else: 
    DEVICE = 'cpu'

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

# def get_transform(SELECTED_AUG,dataset='cifar100'):
#     if dataset == 'cifar100':
#         num_classes = 100
#     elif dataset == 'cifar10':
#         num_classes = 10
#     elif dataset == 'imagenet1K':
#         num_classes = 1000
#     else:
#         print("***** ERROR ERROR ERROR ******")
#         print("Invalid Dataset Selected, Exiting")
#         exit()
#     hparams= {'translate_const': 100, 'img_mean': (124, 116, 104)}
#     normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
#     if SELECTED_AUG == 'cutout':
#         transform = timm.data.random_erasing.RandomErasing(probability=1.0, 
#             min_area=0.02, 
#             max_area=1/3, 
#             min_aspect=0.3, 
#             max_aspect=None,
#             mode='const', 
#             min_count=1, 
#             max_count=None, 
#             num_splits=0, 
#             device=DEVICE)
#     elif SELECTED_AUG == 'mixup':
#         #mixup active if mixup_alpha > 0
#         #cutmix active if cutmix_alpha > 0
#         transform = timm.data.mixup.Mixup(mixup_alpha=1., 
#             cutmix_alpha=0., 
#             cutmix_minmax=None, 
#             prob=1.0, 
#             switch_prob=0.0,
#             mode='batch', 
#             correct_lam=True, 
#             label_smoothing=0.1, 
#             num_classes=num_classes)
#     elif SELECTED_AUG == 'cutmix':
#         transform = timm.data.mixup.Mixup(mixup_alpha=0., 
#             cutmix_alpha=1., 
#             cutmix_minmax=None, 
#             prob=0.8, 
#             switch_prob=0.0,
#             mode='batch', 
#             correct_lam=True, 
#             label_smoothing=0.1, 
#             num_classes=num_classes)
#     elif SELECTED_AUG == 'autoaug':
#         #no searching code;
#         transform = timm.data.auto_augment.auto_augment_transform(config_str='original-mstd0.5',
#         hparams= hparams)
#     elif SELECTED_AUG == 'augmix':
#         transform = timm.data.auto_augment.augment_and_mix_transform(
#             config_str='augmix-m5-w4-d2',
#             hparams=hparams)
#     elif SELECTED_AUG == 'randaug':
#         transform = timm.data.auto_augment.rand_augment_transform( 
#             config_str='rand-m3-n2-mstd0.5',
#             hparams=hparams
#             )
#     elif SELECTED_AUG == 'base':
#         transform = transforms.Compose(
#             [transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(32, padding=4),transforms.ToTensor(),normalize])

#     elif SELECTED_AUG == 'pixmix':
#         transform = None
#     if SELECTED_AUG == 'cutout':
#         pass
#     if SELECTED_AUG in ['randaug','augmix','autoaug']:
#        transform = torchvision.transforms.Compose([transform,torchvision.transforms.ToTensor(),normalize]) 
#     return transform

"""
PixMix Augmentation Code!
"""

def get_ab(beta):
  if np.random.random() < 0.5:
    a = np.float32(np.random.beta(beta, 1))
    b = np.float32(np.random.beta(1, beta))
  else:
    a = 1 + np.float32(np.random.beta(1, beta))
    b = -np.float32(np.random.beta(1, beta))
  return a, b

def add(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 1, img2 * 2 - 1
  out = a * img1 + b * img2
  return (out + 1) / 2

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clip(1e-37) ** b)
  return out / 2

mixings = [add, multiply]
def pixmix(orig, mixing_pic, preprocess,k, beta, severity,use_all_ops): 
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig,severity,use_all_ops))
    else:
        mixed = tensorize(orig)
  
    for _ in range(np.random.randint(k + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig,severity,use_all_ops))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)
    return normalize(mixed)

def augment_input(image,severity, use_all_ops):
    
    #TODO Adjust to take level or severity in Hparams! 
    
    augmentations = timm.data.rand_augment_ops(transforms=['AutoContrast','Equalize','Posterize','Rotate','Solarize','TranslateXRel','TranslateYRel'])
    augmentations_all = timm.data.rand_augment_ops(transforms=['AutoContrast','Equalize','Posterize','Rotate','Solarize','ShearX','ShearY','TranslateXRel','TranslateYRel','Color','Contrast','Brightness','Sharpness'])
    aug_list = augmentations_all if use_all_ops else augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), severity)

class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess,severity,use_all_ops=False):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess
    self.use_all_ops = use_all_ops
    self.severity = severity

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)

def arg_parser():
    parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar100',
        choices=['cifar10', 'cifar100'],
        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument(
        '--data-path',
        type=str,
        default='p/lustre1/trivedi1/vision_data',
        required=False,
        help='Path to CIFAR and CIFAR-C directories')
    parser.add_argument(
        '--mixing-set',
        type=str,
        required=False,
        help='Mixing set directory.')
    parser.add_argument(
        '--use_300k',
        action='store_true',
        help='use 300K random images as aug data'
    )
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='wrn',
        choices=['wrn', 'densenet', 'resnext'],
        help='Choose architecture.')
    # Optimization options
    parser.add_argument(
        '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument(
        '--learning-rate',
        '-lr',
        type=float,
        default=0.1,
        help='Initial learning rate.')
    parser.add_argument(
        '--batch-size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--eval-batch-size', type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument(
        '--decay',
        '-wd',
        type=float,
        default=0.0005,
        help='Weight decay (L2 penalty).')
    # WRN Architecture options
    parser.add_argument(
        '--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=4, type=int, help='Widen factor')
    parser.add_argument(
        '--droprate', default=0.3, type=float, help='Dropout probability')
    # PixMix options
    parser.add_argument(
        '--beta',
        default=3,
        type=int,
        help='Severity of mixing')
    parser.add_argument(
        '--k',
        default=4,
        type=int,
        help='Mixing iterations')
    parser.add_argument(
        '--aug-severity',
        default=3,
        type=int,
        help='Severity of base augmentation operators')
    parser.add_argument(
        '--all-ops',
        '-all',
        action='store_true',
        help='Turn on all augmentation operations (+brightness,contrast,color,sharpness).')
    # Checkpointing options
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
    # Acceleration
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of pre-fetching threads.')

    parser.add_argument(
        '--transform',
        type=str,
        default='augmix',
        help='Choose Transform/Augmentation.')
    args = parser.parse_args()
    return args

class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        # unnormalize
        bx = (bx+1)/2

        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx*2-1

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

def normalize_l2(x):
  """
  Expects x.shape == [N, C, H, W]
  """
  norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
  norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
  return x / norm

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def arg_parser_eval():
    parser = argparse.ArgumentParser(
    description='Evaluates a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar100',
        choices=['cifar10', 'cifar100'],
        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument(
        '--corruption_path',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/CIFAR-100-C/',
        required=False,
        help='Path to CIFAR and CIFAR-C directories')
    parser.add_argument(
        '--perturb_path',
        type=str,
        default='/p/lustre1/trivedi1/vision_data/CIFAR-100-P/',
        required=False,
        help='Path to CIFAR and CIFAR-C directories')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='wrn',
        choices=['wrn', 'densenet', 'resnext'],
        help='Choose architecture.')
    # WRN Architecture options
    parser.add_argument(
        '--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=4, type=int, help='Widen factor')
    parser.add_argument(
        '--droprate', default=0.3, type=float, help='Dropout probability')
    parser.add_argument(
        '--resume',
        '-r',
        type=str,
        default='',
        help='Checkpoint path for resume / test.')
    parser.add_argument('--eval_all', action='store_true', help='Performs all ML Safety evaluation.')
    parser.add_argument('--eval_C', action='store_true', help='Eval on Corruptions.')
    parser.add_argument('--eval_P', action='store_true', help='Eval on Perturbation.')
    parser.add_argument('--eval_A', action='store_true', help='Eval on Adversarial Examples.')
    parser.add_argument('--eval_O',action='store_true',help='Eval on OOD Datasets')
    parser.add_argument('--eval_Cal',action='store_true',help='Eval Calibration Error')
    parser.add_argument('--eval_Clean',action='store_true',help='Clean Accuracy')
    parser.add_argument('--use_xent',action='store_true',help='Use CrossEntropy in the OOD Eval.')
    parser.add_argument("--model_str",default='',help='Provide an identifier for the checkpoint')
    parser.add_argument(
        '--transform',
        type=str,
        default='augmix',
        help='Choose Transform/Augmentation.')
    # data loader args
    parser.add_argument(
        '--eval_batch_size', default=32, type=int, help='Eval Batchsize')
    parser.add_argument(
        '--num_workers',default=4, type=int, help='Num Workers')
    parser.add_argument(
        '--num_to_avg',default=1, type=int, help='Num to Avg.')
    parser.add_argument(
        '--prefetch',action='store_true', help='Prefetch Ood Loader')
 
    args = parser.parse_args()
    return args 

def get_ood_scores(net, loader, ood_num_examples, args, in_dist=False):
    
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy() 
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.eval_batch_size and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        #in_score, right_score, wrong_score 
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()
    
def get_and_print_results(net,ood_loader,ood_num_examples, in_score, args):

    aurocs, auprs, fprs = [], [], []
    for _ in range(args.num_to_avg):
        out_score = get_ood_scores(net=net, 
        loader=ood_loader, 
        ood_num_examples=ood_num_examples,
        args=args,
        in_dist=False)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)

    if args.num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.save_name)
    else:
        print_measures_ood(auroc, aupr, fpr, args.save_name)
    return auroc, aupr, fpr

recall_level_default = 0.95
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def print_measures_ood(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(aurocs, auprs, fprs, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))


def show_performance_comparison(pos_base, neg_base, pos_ours, neg_ours, baseline_name='Baseline',
                                method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos_base: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg_base: 0's class scores generated by the baseline
    '''
    auroc_base, aupr_base, fpr_base = get_measures(pos_base[:], neg_base[:], recall_level)
    auroc_ours, aupr_ours, fpr_ours = get_measures(pos_ours[:], neg_ours[:], recall_level)

    print('\t\t\t' + baseline_name + '\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
        int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours))
    print('AUROC:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * auroc_base, 100 * auroc_ours))
    print('AUPR:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * aupr_base, 100 * aupr_ours))
    # print('FDR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
    #     int(100 * recall_level), 100 * fdr_base, 100 * fdr_ours))


def flip_prob(predictions, difficulty, noise_perturbation=False):
    result = 0
    step_size = 1 if noise_perturbation else difficulty

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result
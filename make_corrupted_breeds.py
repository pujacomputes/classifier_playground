# -*- coding: utf-8 -*-
import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
import sys
from PIL import Image


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
from torch.utils.data import Dataset
from breeds_helper import make_living17, make_entity13, make_entity30, make_nonliving26
warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# def fgsm(x, source_net, severity=1):
#     c = [8, 16, 32, 64, 128][severity - 1]

#     x = V(x, requires_grad=True)
#     logits = source_net(x)
#     source_net.zero_grad()
#     loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
#     loss.backward()

#     return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


# def barrel(x, severity=1):
#     c = [(0,0.03,0.03), (0.05,0.05,0.05), (0.1,0.1,0.1),
#          (0.2,0.2,0.2), (0.1,0.3,0.6)][severity - 1]
#
#     output = BytesIO()
#     x.save(output, format='PNG')
#
#     x = WandImage(blob=output.getvalue())
#     x.distort('barrel', c)
#
#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)
#
#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    prefix = '/p/lustre3/trivedi1/CreateCorrupted/frost_img'
    filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg'][idx]
    frost = cv2.imread("{}/{}".format(prefix,filename))
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////


# /////////////// Further Setup ///////////////
# /////////////// Data Loader ///////////////


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


"""
We need to load create corrupted versions of the data
that correspond to the domainnet test file. This means,
we need to keep the same path name when saving to the new location.
"""


BREEDS_SPLITS_TO_FUNC = {
    'entity13': make_entity13,
    'entity30': make_entity30,
    'living17': make_living17,
    'nonliving26': make_nonliving26,
}

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

SPLITS = ['train', 'val']

MIN_NUM_TRAIN_PER_CLASS = 100

NUM_VAL_PER_CLASS = 50

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def get_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def get_image_paths_breeds_class(class_dir, breeds_class):
    image_paths_breeds_class = []
    for root, _, fnames in sorted(os.walk(class_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions=IMG_EXTENSIONS):
                path = os.path.join(root, fname)
                image_paths_breeds_class.append((path, breeds_class))
    return image_paths_breeds_class

def get_image_paths_by_class(data_dir, idx_to_class_id, subclasses, split):
    image_paths_and_class = []
    for idx in range(len(subclasses)):
        for subclass in subclasses[idx]:
            subclass_image_paths_breeds_class = get_image_paths_breeds_class(
                data_dir + '/' + idx_to_class_id[subclass] + '/', idx)
            image_paths_and_class.extend(subclass_image_paths_breeds_class)
            # print(data_dir + '/' + idx_to_class_id[subclass] + '/', len(subclass_image_names))
            if split == 'train':
                assert(len(subclass_image_paths_breeds_class) >= MIN_NUM_TRAIN_PER_CLASS)
            else:
                assert(len(subclass_image_paths_breeds_class) == NUM_VAL_PER_CLASS)
    return image_paths_and_class

class Breeds(Dataset):
    def __init__(self, root, breeds_name,method,severity,
                 info_dir='/usr/workspace/trivedi1/vision_data/breeds_info',
                 source=True, target=False, split='train', transform=None):
        super().__init__()
        if breeds_name not in BREEDS_SPLITS_TO_FUNC.keys():
            raise ValueError(f'breeds_name must be in {BREEDS_SPLITS_TO_FUNC.keys()} but was {breeds_name}')
        if split not in SPLITS:
            raise ValueError(f'split must be in {SPLITS} but was {split}')
        if not source and not target:
            raise ValueError('At least one of "source" and "target" must be True!')

        self._breeds_name = breeds_name
        self._source = source
        self._split = split
        self._transform = transform
        self._info_dir = info_dir
        self._data_dir = root + '/' + split
        self._idx_to_class_id, self._class_to_idx = get_classes(self._data_dir)
        breeds_func = BREEDS_SPLITS_TO_FUNC[breeds_name]
        self._superclasses, self._subclass_split, self._label_map = breeds_func(self._info_dir, split="rand")
        self._subclasses = []

        if source:
            self._subclasses.extend(self._subclass_split[0])
        if target:
            self._subclasses.extend(self._subclass_split[1])

        self._image_paths_by_class = get_image_paths_by_class(
            self._data_dir, self._idx_to_class_id, self._subclasses, split)

        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.228, 0.224, 0.225]
        self.method = method
        self.severity = severity
        print("\t=> Method: ",self.method)
        print("\t=> Severity: ",self.method)

    def __getitem__(self, i):
        path, y = self._image_paths_by_class[i]
        file_name = path.split("/usr/workspace/trivedi1/vision_data/ImageNet/")[1]
        x = Image.open(path)
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
            x = self.method(x,self.severity)
        save_path = '/usr/workspace/wsa/trivedi1/vision_data/Living17-Corrupted/{method}/{severity}/{img_path}'.format(method = self.method.__name__,
            severity= self.severity,
            img_path = file_name)
        Image.fromarray(np.uint8(x)).save(save_path, quality=85, optimize=True)
        return y

    def __len__(self) -> int:
        return len(self._image_paths_by_class)

    def get_num_classes(self):
        return len(self._idx_to_class_id)

def save_distorted(method=gaussian_noise):
    for severity in range(1, 6):
        print(method.__name__, severity)
       
        distorted_dataset= Breeds(root='/usr/workspace/trivedi1/vision_data/ImageNet', 
            breeds_name='living17', 
            method=method,
            severity=severity,
            info_dir='/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified',
            source=True, target=False, split='val', transform=trn.Resize((224,224)))
        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=100, shuffle=False, num_workers=4)

        for _ in distorted_dataset_loader: continue

# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create Corrupted")
    parser.add_argument("--env", type=str, default='sketch')
    parser.add_argument('--corruptions', nargs='*', help='Select Corruptions', required=False)
    args = parser.parse_args()

    print('\nUsing Living17 data')
    print("\t Corruptions: ",args.corruptions) 
    d = collections.OrderedDict()
    d['Gaussian Noise'] = gaussian_noise
    d['Shot Noise'] = shot_noise
    d['Impulse Noise'] = impulse_noise
    d['Defocus Blur'] = defocus_blur
    d['Glass Blur'] = glass_blur
    d['Motion Blur'] = motion_blur
    d['Zoom Blur'] = zoom_blur
    d['Snow'] = snow
    d['Frost'] = frost
    d['Fog'] = fog
    d['Brightness'] = brightness
    d['Contrast'] = contrast
    d['Elastic'] = elastic_transform
    d['Pixelate'] = pixelate
    d['JPEG'] = jpeg_compression

    d['Speckle Noise'] = speckle_noise
    d['Gaussian Blur'] = gaussian_blur
    d['Spatter'] = spatter
    d['Saturate'] = saturate

    #premake directories to avoid dataloader conflicts
    corrupt_funcs = ['Gaussian Noise', 
        'Shot Noise',
        'Impulse Noise',
        'Defocus Blur',
        'Glass Blur',
        'Motion Blur',
        'Zoom Blur',
        'Snow',
        'Frost',
        'Fog',
        'Brightness',
        'Contrast',
        'Elastic',
        'Pixelate',
        'JPEG',
        'Speckle Noise',
        'Gaussian Blur',
        'Spatter',
        'Saturate']

    """
    Create the folder structure.
    """
    bdataset = Breeds(root='/usr/workspace/trivedi1/vision_data/ImageNet', 
        breeds_name='living17', 
        method='spatter',
        severity=1,
        info_dir='/usr/workspace/trivedi1/vision_data/BREEDS-Benchmarks/imagenet_class_hierarchy/modified',
        source=True, target=False, split='val', transform=None)

    class_ints = np.array(bdataset._subclasses).flatten().tolist()
    class_idx = [bdataset._idx_to_class_id[k] for k in class_ints]

    # for method_name in d.keys():
    #     for severity in range(1,6):
    #         for c in class_idx:
    #             save_path = '/usr/workspace/wsa/trivedi1/vision_data/Living17-Corrupted/{method}/{severity}/val/{class_name}'.format(env=args.env,
    #                 method = d[method_name].__name__, 
    #                 severity= severity,
    #                 class_name =c)
    #             print("=> Folder Path: ",save_path)
    #             if not os.path.exists(save_path):
    #                 os.makedirs(save_path)
    for method_name in args.corruptions:
        print("=> Method Name: ",method_name)
        save_distorted(method=d[method_name])

if __name__ == '__main__':
    main()
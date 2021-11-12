import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image
from cv2.cv2 import Canny, GaussianBlur
from omegaconf import OmegaConf

from taming.data.base import ImagePaths
    
class WikiartBase(Dataset):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class WikiartTrain(WikiartBase):
    def __init__(self, size=256, img_list_file="datasets/wikiart_train.txt"):
        super().__init__()
        with open(img_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class WikiartTest(WikiartBase):
    def __init__(self, size=256, img_list_file="datasets/wikiart_test.txt"):
        super().__init__()
        with open(img_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        
def imscale(x, factor, keepshapes=False, keepmode="bicubic"):
    if factor is None or factor==1:
        return x

    dtype = x.dtype
    assert dtype in [np.float32, np.float64]
    assert x.min() >= -1
    assert x.max() <= 1

    keepmode = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC}[keepmode]

    lr = (x+1.0)*127.5
    lr = lr.clip(0,255).astype(np.uint8)
    lr = Image.fromarray(lr)

    h, w, _ = x.shape
    nh = h//factor
    nw = w//factor
    assert nh > 0 and nw > 0, (nh, nw)

    lr = lr.resize((nw,nh), Image.BICUBIC)
    if keepshapes:
        lr = lr.resize((w,h), keepmode)
    lr = np.array(lr)/127.5-1.0
    lr = lr.astype(dtype)

    return lr

class WikiartScale(Dataset):
    def __init__(self, size=None, crop_size=None, random_crop=False,
                 up_factor=None, hr_factor=None, keep_mode="bicubic", x_flip=False):
        self.base = self.get_base()

        self.size = size
        self.crop_size = crop_size if crop_size is not None else self.size
        self.random_crop = random_crop
        self.up_factor = up_factor
        self.hr_factor = hr_factor
        self.keep_mode = keep_mode
        self.x_flip = x_flip

        transforms = list()

        if self.size is not None and self.size > 0:
            rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            self.rescaler = rescaler
            transforms.append(rescaler)

        if self.crop_size is not None and self.crop_size > 0:
            if len(transforms) == 0:
                self.rescaler = albumentations.SmallestMaxSize(max_size = self.crop_size)

            if not self.random_crop:
                cropper = albumentations.CenterCrop(height=self.crop_size,width=self.crop_size)
            else:
                cropper = albumentations.RandomCrop(height=self.crop_size,width=self.crop_size)
            transforms.append(cropper)

        if self.x_flip:
            flipper = albumentations.HorizontalFlip(p=1.0)
            transforms.append(flipper)

        if len(transforms) > 0:
            if self.up_factor is not None:
                additional_targets = {"lr": "image"}
            else:
                additional_targets = None
            self.preprocessor = albumentations.Compose(transforms,
                                                       additional_targets=additional_targets)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = example["image"]
        # adjust resolution
        image = imscale(image, self.hr_factor, keepshapes=False)
        h,w,c = image.shape
        if self.crop_size and min(h,w) < self.crop_size:
            # have to upscale to be able to crop - this just uses bilinear
            image = self.rescaler(image=image)["image"]
        if self.up_factor is None:
            image = self.preprocessor(image=image)["image"]
            example["image"] = image
        else:
            lr = imscale(image, self.up_factor, keepshapes=True,
                         keepmode=self.keep_mode)

            out = self.preprocessor(image=image, lr=lr)
            example["image"] = out["image"]
            example["lr"] = out["lr"]

        return example

class WikiartScaleTrain(WikiartScale):
    def __init__(self, random_crop=True, **kwargs):
        super().__init__(random_crop=random_crop, **kwargs)

    def get_base(self):
        return WikiartBase()

class WikiartScaleValidation(WikiartScale):
    def get_base(self):
        return WikiartBase()

from skimage.feature import canny
from skimage.color import rgb2gray
from scipy import ndimage as ndi

def rgba_to_edge(x):
    assert x.dtype == np.uint8
    assert len(x.shape) == 3 and x.shape[2] == 4
    y = x.copy()
    y.dtype = np.float32
    y = y.reshape(x.shape[:2])
    y = np.invert(y)
    return np.ascontiguousarray(y)

class WikiartEdges(WikiartScale):
    def __init__(self, up_factor=1, **kwargs):
        super().__init__(up_factor=1, **kwargs)
        
    def __getitem__(self, i):
        example = self.base[i]
        image = example["image"]
        h,w,c = image.shape
        if self.crop_size and min(h,w) < self.crop_size:
            # have to upscale to be able to crop - this just uses bilinear
            image = self.rescaler(image=image)["image"]

        lr = canny(rgb2gray(image), 1.75, 0.15, 0.75)
        lr = 1.0 - lr.astype(np.float32)

        out = self.preprocessor(image=image, lr=lr)
        example["image"] = out["image"]
        example["lr"] = out["lr"]

        return example  

class WikiartEdgesTrain(WikiartEdges):
    def __init__(self, size, img_list_file, random_crop=True, x_flip=True, **kwargs):
        self.size = size
        self.img_list_file = img_list_file
        super().__init__(random_crop=random_crop, x_flip=x_flip, **kwargs)

    def get_base(self):
        return WikiartTrain()

class WikiartEdgesTest(WikiartEdges):
    def __init__(self, size, img_list_file, **kwargs):
        self.size = size
        self.img_list_file = img_list_file
        super().__init__(**kwargs)

    def get_base(self):
        return WikiartTest()
    
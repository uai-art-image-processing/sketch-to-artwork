import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image
from cv2.cv2 import Canny
from omegaconf import OmegaConf

from taming.data.base import ImagePaths

# def rgba_to_edge(x):
#     assert x.dtype == np.uint8
#     assert len(x.shape) == 3 and x.shape[2] == 4
#     y = x.copy()
#     y.dtype = np.float32
#     y = y.reshape(x.shape[:2])
#     return np.ascontiguousarray(y)

# class Wikiart(Dataset):
#     def __init__(self, size, images_list_file, *args, **kwargs):
#         super().__init__()
#         with open(images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, random_crop=False)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         example = self.data[i]
#         return example
    
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

# class WikiartTrain(WikiartBase):
#     def __init__(self, size, training_images_list_file="data/wikiart_train.txt"):
#         super().__init__()
#         with open(training_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, random_crop=False)


# class WikiartTest(WikiartBase):
#     def __init__(self, size, test_images_list_file="data/wikiart_test.txt"):
#         super().__init__()
#         with open(test_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        

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
        
#     def preprocess_edge(self, rgb):
#         edge = rgb_to_edge(rgba)
#         edge = (edge - edge.min())/max(1e-8, edge.max()-edge.min())
#         edge = 2.0*edge-1.0
#         return edge

#     def get_edge(image, sigma=2, low_threshold=0.1, high_threshold=0.2):
#         # Edge preprocessing
#         out = canny(rgb2gray(image), sigma, low_threshold, high_threshold)
#         out = 1.0 - out.astype(np.float32)
#         # lr = lr[:,:,None][:,:,[0,0,0]]
#         return out
        
    def __getitem__(self, i):
        example = self.base[i]
        image = example["image"]
        h,w,c = image.shape
        if self.crop_size and min(h,w) < self.crop_size:
            # have to upscale to be able to crop - this just uses bilinear
            image = self.rescaler(image=image)["image"]
        # lr = self.get_edge(image, 3, 0.25, 0.75)
        lr = canny(rgb2gray(image), 2)
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

# class EdgePaths(ImagePaths):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#     def preprocess_image(self, image_path):
#         image = Image.open(image_path)
#         if not image.mode == "L":
#             image = image.convert("L")
#         image = np.array(image).astype(np.uint8)
#         image = self.preprocessor(image=image)["image"]
#         image = (image/127.5 - 1.0).astype(np.float32)
#         return image

# class BaseWithDepth(Dataset):
#     DEFAULT_DEPTH_ROOT="data/wikiart_edges"

#     def __init__(self, config=None, size=None, random_crop=False,
#                  crop_size=None, root=None, split="train"):
#         self.config = config
#         self.base_dset = self.get_base_dset()
#         self.preprocessor = get_preprocessor(
#             size=size,
#             crop_size=crop_size,
#             random_crop=random_crop,
#             additional_targets={"edge": "image"})
#         self.crop_size = crop_size
#         if self.crop_size is not None:
#             self.rescaler = albumentations.Compose(
#                 [albumentations.SmallestMaxSize(max_size = self.crop_size)],
#                 additional_targets={"edge": "image"})
#         if root is not None:
#             self.DEFAULT_DEPTH_ROOT = root
#         self.split = split

#     def __len__(self):
#         return len(self.base_dset)

#     def preprocess_edge(self, path):
#         img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 
#                     cv2.IMREAD_UNCHANGED)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.GaussianBlur(img, (5,5), 0)
#         edge = cv2.Canny(img, 75, 250)
#         return cv2.bitwise_not(edge)

#     def __getitem__(self, i):
#         e = self.base_dset[i]
#         e["edge"] = self.preprocess_edge(self.get_depth_path(e))
#         # up if necessary
#         h,w,c = e["image"].shape
#         if self.crop_size and min(h,w) < self.crop_size:
#             # have to upscale to be able to crop - this just uses bilinear
#             out = self.rescaler(image=e["image"], depth=e["edge"])
#             e["image"] = out["image"]
#             e["edge"] = out["edge"]
#         transformed = self.preprocessor(image=e["image"], depth=e["edge"])
#         e["image"] = transformed["image"]
#         e["edge"] = transformed["edge"]
#         return e

#     def get_base_dset(self):
#         return Wikiart()

#     def get_depth_path(self, e):
#         fid = os.path.splitext(e["relpath"])[0]+".jpg"
#         fid = os.path.join(self.DEFAULT_DEPTH_ROOT, self.split, fid)
#         return fid


# class WikiartTrainWithDepth(BaseWithDepth):
#     # default to random_crop=True
#     def __init__(self, random_crop=True, sub_indices=None, **kwargs):
#         self.sub_indices = sub_indices
#         super().__init__(random_crop=random_crop, **kwargs)

#     def get_base_dset(self):
#         return Wikiart()

#     def get_depth_path(self, e):
#         fid = os.path.splitext(e["relpath"])[0]+".png"
#         fid = os.path.join(self.DEFAULT_DEPTH_ROOT, "train", fid)
#         return fid

# class WikiartTestWithDepth(BaseWithDepth):
#     def __init__(self, sub_indices=None, **kwargs):
#         self.sub_indices = sub_indices
#         super().__init__(**kwargs)

#     def get_base_dset(self):
#         if self.sub_indices is None:
#             return WikiartTest()
#         else:
#             return WikiartTest({"sub_indices": self.sub_indices})

#     def get_depth_path(self, e):
#         fid = os.path.splitext(e["relpath"])[0]+".png"
#         fid = os.path.join(self.DEFAULT_DEPTH_ROOT, "val", fid)
#         return fid
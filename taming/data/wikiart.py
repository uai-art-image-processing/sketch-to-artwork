import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image

from taming.data.base import ImagePaths

def rgba_to_depth(x):
    assert x.dtype == np.uint8
    assert len(x.shape) == 3 and x.shape[2] == 4
    y = x.copy()
    y.dtype = np.float32
    y = y.reshape(x.shape[:2])
    return np.ascontiguousarray(y)

class EdgePaths(ImagePaths):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "L":
            image = image.convert("L")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

class WikiartBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class WikiartEdgesTrain(WikiartBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = EdgePaths(paths=paths, size=size, random_crop=False)


class WikiartEdgesTest(WikiartBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = EdgePaths(paths=paths, size=size, random_crop=False)

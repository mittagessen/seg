import torch.utils.data as data
import torch

import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as tf
import os
from PIL import Image
from scipy.ndimage import maximum_filter


class BaselineSet(data.Dataset):
    def __init__(self, imgs, augment=True):
        super(BaselineSet, self).__init__()
        self.imgs = imgs
        self.targets = [os.path.splitext(x)[0] + '.png' for x in imgs]
        self.augment = augment

    def __getitem__(self, idx):
        input = Image.open(self.imgs[idx]).convert('RGB')
        target = Image.open(self.targets[idx])
        return self.transform(input, target)

    def transform(self, image, target):
        resize = transforms.Resize(1200)
        jitter = transforms.ColorJitter()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        res = tf.to_tensor(resize(image))
        image = norm(res)
        #image = jitter(res)
        target = resize(target)
        #mask = np.expand_dims(np.maximum(maximum_filter(target, (30, 150)), 0.0), 0)
        return image, tf.to_tensor(target)

    def __len__(self):
        return len(self.imgs)


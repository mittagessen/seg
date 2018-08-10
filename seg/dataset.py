import torch.utils.data as data
import torch

import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as tf
import os
from PIL import Image


class BaselineSet(data.Dataset):
    def __init__(self, imgs, augment=True):
        super(BaselineSet, self).__init__()
        self.imgs = imgs
        self.targets = [os.path.splitext(x)[0] + '.png' for x in imgs]
        self.augment = augment

    def __getitem__(self, idx):
        input = Image.open(self.imgs[idx]).convert('1')
        target = Image.open(self.targets[idx])
        return self.transform(input, target)

    def transform(self, image, target):
        resize = transforms.Resize(1200)
        jitter = transforms.ColorJitter()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        res = resize(image)
        image = res
        #image = jitter(res)
        target = resize(target)

        if self.augment:
            if np.random.random() > 0.5:
                image = tf.hflip(image)
                target = tf.hflip(target)

            angle = np.random.uniform(-10, 10)
            image = tf.rotate(image, angle, resample=Image.BICUBIC)
            target = tf.rotate(target, angle, resample=Image.NEAREST)
        image = tf.to_tensor(image)

        return image, tf.to_tensor(target), tf.to_tensor(res)

    def __len__(self):
        return len(self.imgs)


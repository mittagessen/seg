import torch.utils.data as data
import torch

import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as tf
import os
from PIL import Image
from scipy.ndimage import maximum_filter, find_objects
from skimage.morphology import convex_hull_image


class BaselineSet(data.Dataset):
    def __init__(self, imgs, augment=True):
        super(BaselineSet, self).__init__()
        self.imgs = [x[:-10] for x in imgs]
        self.augment = augment

    def __getitem__(self, idx):
        im = self.imgs[idx]
        target = Image.open('{}.seeds.png'.format(im))
        orig = Image.open('{}.plain.png'.format(im)).convert('RGB')

        return self.transform(orig, target)

    def transform(self, image, target):
        resize = transforms.Resize(900)
        jitter = transforms.ColorJitter()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        res = tf.to_tensor(resize(image))
        image = norm(res)
        #image = jitter(res)
        target = Image.fromarray(((np.array(target) > 0) * 255).astype('uint8'))
        target = resize(target)
        return image, tf.to_tensor(target)

    def __len__(self):
        return len(self.imgs)


class DilationSet(data.Dataset):
    def __init__(self, imgs, augment=True, dilation=10):
        super(DilationSet, self).__init__()
        self.imgs = [x[:-10] for x in imgs]
        self.augment = augment
        self.items = 0
        self.dilation = dilation
        for im in self.imgs:
            self.items += int(np.array(Image.open('{}.seeds.png'.format(im))).max())

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.imgs))
        im = self.imgs[idx]
        seeds = Image.open('{}.seeds.png'.format(im))
        lines = Image.open('{}.lines.png'.format(im))
        orig = Image.open('{}.plain.png'.format(im)).convert('RGB')
        return self.transform(seeds, lines, orig)

    def transform(self, seeds, lines, orig):
        resize = transforms.Resize(900)
        resize_nearest = transforms.Resize(900, Image.NEAREST)

        seeds = resize_nearest(seeds)
        lines = resize_nearest(lines)
        orig = resize(orig)

        # extract bounding box of random line
        seeds = np.array(seeds)
        idx = np.random.randint(seeds.max()+1)
        bbox = find_objects(seeds == idx)[0]
        bbox = (slice(bbox[0].start - self.dilation, bbox[0].stop + self.dilation, bbox[0].step),
                slice(bbox[1].start - self.dilation, bbox[1].stop + self.dilation, bbox[1].step))
        seeds = (seeds[bbox] == idx) * 255
        seeds = tf.to_tensor(np.expand_dims(seeds, 2)).float()
        # expanded target
        target = (np.array(lines)[bbox] == idx) * 255
        #target = convex_hull_image(target) * 255
        target = tf.to_tensor(np.expand_dims(target, 2)).float()
        # original image
        orig = tf.to_tensor(np.array(orig)[bbox])

        return torch.cat([seeds, orig], 0), target

    def __len__(self):
        return self.items

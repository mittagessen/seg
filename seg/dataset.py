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
        self.cmap = [(1, (0, 0, 255)), (2, (255, 0, 0)), (3, (255, 255, 0))]

    def __getitem__(self, idx):
        input = Image.open(self.imgs[idx]).convert('RGB')
        target = Image.open(self.targets[idx]).convert('RGB')
        return self.transform(input, target)

    def get_target_weights(self):
        """
        Calculates the proportion of each class in the training set.
        """
        tot = 0
        cnts = torch.zeros(4)
        for im in self.targets:
            target = Image.open(im)
            target = np.array(target)
            for v, m in self.cmap:
                cnts[v] += np.count_nonzero(np.all(target == m, axis=1))
            tot += target.size
        return 1 / (cnts / tot)

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

        target = np.array(target)
        l = np.zeros(target.shape[:2], 'i')
        for v, m in self.cmap:
            l[np.all(target == m, axis=-1)] = v
        target = torch.LongTensor(l)
        return norm(image), target, tf.to_tensor(res)

    def __len__(self):
        return len(self.imgs)


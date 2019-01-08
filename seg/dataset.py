import torch.utils.data as data
import torch

import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as tf
import os
from PIL import Image
from scipy.ndimage import maximum_filter, find_objects
from skimage.morphology import convex_hull_image
from torch.nn.utils.rnn import pad_sequence
from seg import degrade

class BaselineSet(data.Dataset):
    def __init__(self, imgs, augment=True):
        super(BaselineSet, self).__init__()
        self.imgs = [x[:-10] for x in imgs]

    def __getitem__(self, idx):
        im = self.imgs[idx]
        target = Image.open('{}.seeds.png'.format(im))
        orig = Image.open('{}.plain.png'.format(im))

        return self.transform(orig, target)

    def transform(self, image, target):
        image = image.convert('L')
        resize = transforms.Resize(900)

        image = tf.to_tensor(resize(image)).squeeze(0)
        target = tf.to_tensor(resize(target)).squeeze(0)

        noise = degrade.bounded_gaussian_noise(image.shape, np.random.randint(5), 3.0)
        image = degrade.distort_with_noise(image, noise)
        target = degrade.distort_with_noise(target, noise)

        if np.random.randint(2):
            image = 1-degrade.printlike_multiscale(image, blur=1)

        target = Image.fromarray(((target > 0) * 255).astype('uint8'))
        image = Image.fromarray((image * 255).astype('uint8')).convert('RGB')

        return tf.to_tensor(image), tf.to_tensor(np.expand_dims(target, 2))

    def __len__(self):
        return len(self.imgs)

    def get_target_weights(self):
        tot = 0
        cnt = 0
        for im in self.imgs:
            ar = np.array(Image.open('{}.seeds.png'.format(im)))
            tot += ar.size
            cnt += np.count_nonzero(ar)
        return torch.tensor(tot / cnt)

def dilate_collate(batch):
    """
    Pads all images in batch to the largest image size and assembles a tensor.
    """
    batch = sorted(batch, key=lambda x: x[0].shape[2], reverse=True)
    seq_len = torch.IntTensor([l[0].shape[2] for l in batch])
    inp = pad_sequence([x[0].transpose(0, 2) for x in batch]).permute(1, 3, 2, 0).contiguous()
    target = pad_sequence([x[1].transpose(0, 2) for x in batch]).permute(1, 3, 2, 0).contiguous()
    return seq_len, inp, target

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
        bbox = (slice(bbox[0].start - self.dilation, bbox[0].start + self.dilation, bbox[0].step),
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

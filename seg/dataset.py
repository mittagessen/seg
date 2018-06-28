import torch.utils.data as data
import torch

import numpy as np
from torchvision import transforms
import os
from PIL import Image

default_input_transforms = transforms.Compose([transforms.Resize(1200), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
default_target_transforms = transforms.Compose([transforms.Resize(1200)])

class BaselineSet(data.Dataset):
    def __init__(self, imgs, input_transforms=default_input_transforms, target_transforms=default_target_transforms):
        super(BaselineSet, self).__init__()
        self.imgs = imgs
        self.targets = [os.path.splitext(x)[0] + '.png' for x in imgs]
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

    def __getitem__(self, idx):
        input = Image.open(self.imgs[idx]).convert('RGB')
        input = self.input_transforms(input)
        target = Image.open(self.targets[idx])
        target = self.target_transforms(target)
        target = np.array(target)[:,:,2]
        l = np.zeros(target.shape, 'i')
        vals = [(3, 0b1000), (2, 0b0100), (1, 0b0010)]
        for v, m in vals:
            l[np.bitwise_and(m, target) != 0] = v
        return input, torch.LongTensor(l)

    def __len__(self):
        return len(self.imgs)


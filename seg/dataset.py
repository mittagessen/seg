import torch.utils.data as data

from torchvision import transforms
import os
from PIL import Image

default_input_transforms = transforms.Compose([transforms.Resize(1200), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
default_target_transforms = transforms.Compose([transforms.Resize(1200), transforms.ToTensor()])

class BaselineSet(data.Dataset):
    def __init__(self, imgs, input_transforms=default_input_transforms, target_transforms=default_target_transforms):
        super(BaselineSet, self).__init__()
        self.imgs = imgs
        self.targets = [os.path.splitext(x)[0] + '.lines.png' for x in imgs]
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

    def __getitem__(self, idx):
        input = Image.open(self.imgs[idx]).convert('RGB')
        input = self.input_transforms(input)
        target = Image.open(self.targets[idx]).convert('1')
        target = self.target_transforms(target)
        return input, target

    def __len__(self):
        return len(self.imgs)


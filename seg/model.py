import torch

from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torch import autograd

from torch import nn

from PIL import Image

# NCHW
t = transforms.Compose([transforms.Resize(1200), transforms.Lambda(lambda x: x.convert('RGB')), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _wi(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.LSTM):
        for p in m.parameters():
            # weights
            if p.data.dim() == 2:
                torch.nn.init.orthogonal_(p.data)
            # initialize biases to 1 (jozefowicz 2015)
            else:
                torch.nn.init.constant_(p.data[len(p)//4:len(p)//2], 1.0)
    elif isinstance(m, torch.nn.GRU):
        for p in m.parameters():
            torch.nn.init.orthogonal_(p.data)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

class ReNet(nn.Module):
    """
    Recurrent block from ReNet.

    Performs a horizontal pass over input features, followed by a vertical pass
    over the output of the first pass.
    """
    def __init__(self, in_channels, out_channels, bidi=True):
        super(ReNet, self).__init__()
        self.bidi = bidi
        self.hidden_size = out_channels
        self.output_size = out_channels if not self.bidi else 2*out_channels
        self.hrnn = nn.LSTM(in_channels, self.hidden_size, batch_first=True, bidirectional=bidi)
        self.vrnn = nn.LSTM(self.output_size, out_channels, batch_first=True, bidirectional=bidi)

    def forward(self, inputs):
        # horizontal pass
        # NCHW -> HNWC
        inputs = inputs.permute(2, 0, 3, 1)
        siz = inputs.size()
        # HNWC -> (H*N)WC
        inputs = inputs.contiguous().view(-1, siz[2], siz[3])
        # (H*N)WO
        o, _ = self.hrnn(inputs)
        # resize to HNWO
        o = o.view(siz[0], siz[1], siz[2], self.output_size)
        # vertical pass
        # HNWO -> WNHO
        o = o.transpose(0, 2)
        # (W*N)HO
        o = o.view(-1, siz[0], self.output_size)
        # (W*N)HO'
        o, _ = self.vrnn(o)
        # (W*N)HO' -> WNHO'
        o = o.view(siz[2], siz[1], siz[0], self.output_size)
        # WNHO' -> NO'HW
        return o.permute(1, 3, 2, 0)


class ConvReNet(nn.Module):
    """
    Baseline labelling network.
    """
    def __init__(self, cls, refine_encoder=False):
        super(ConvReNet, self).__init__()
        self.cls = cls
        r = 3
        self.label = nn.Sequential(nn.Conv2d(3, 16, r, padding=r//2),
                                   nn.Dropout(0.1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(16, 32, r, padding=r//2),
                                   nn.Dropout(0.1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(32, 64, r, padding=r//2),
                                   nn.Dropout(0.1),
                                   nn.ReLU(),
                                   ReNet(64, 32),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(64, 32, 1),
                                   nn.Dropout(0.1),
                                   nn.ReLU(),
                                   ReNet(32, 32),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(64, 1, 1),
                                   nn.Sigmoid())
        self.init_weights()

    def forward(self, inputs):
        siz = inputs.size()
        o = self.label(inputs)
        return F.interpolate(o, size=(siz[2], siz[3]))

    def init_weights(self):
        self.label.apply(_wi)


class UnetDecoder(nn.Module):
    """
    U-Net decoder block with a convolution before upsampling.
    """
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(inter_channels, out_channels, 3, padding=1, stride=2)

    def forward(self, x, output_size):
        x = F.relu(self.conv(x))
        return F.relu(self.deconv(x, output_size=output_size))


class ResUNet(nn.Module):
    """
    ResNet-34 encoder + U-Net decoder
    """
    def __init__(self, cls=1, refine_encoder=False):
        super(ResUNet, self).__init__()
        self.cls = cls
        # squeezenet feature extractor
        self.resnet = models.resnet34(pretrained=True)
        if not refine_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        # operating on map_4
        self.upsample_4 = UnetDecoder(256, 256, 128)
        # operating on cat(map_3, upsample_5(map_4))
        self.upsample_3 = UnetDecoder(256, 128, 64)
        self.upsample_2 = UnetDecoder(128, 64, 64)
        self.upsample_1 = UnetDecoder(128, 64, 64)
        self.squash = nn.Conv2d(64, cls, kernel_size=1)

        self.nonlin = nn.Sigmoid()
        self.init_weights()

    def forward(self, inputs):
        siz = inputs.size()
        x = self.resnet.conv1(inputs)
        x = self.resnet.bn1(x)
        map_1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        map_2 = self.resnet.layer1(x)
        map_3 = self.resnet.layer2(map_2)
        map_4 = self.resnet.layer3(map_3)

        # upsample concatenated maps
        map_4 = self.dropout(self.upsample_4(map_4, output_size=map_3.size()))
        map_3 = self.dropout(self.upsample_3(torch.cat([map_3, map_4], 1), output_size=map_2.size()))
        map_2 = self.dropout(self.upsample_2(torch.cat([map_2, map_3], 1), output_size=map_1.size()))
        map_1 = self.dropout(self.upsample_1(torch.cat([map_1, map_2], 1), output_size=map_1.size()[:2] + siz[2:]))
        return self.nonlin(self.squash(map_1))

    def init_weights(self):
        self.upsample_4.apply(_wi)
        self.upsample_3.apply(_wi)
        self.upsample_2.apply(_wi)
        self.upsample_1.apply(_wi)
        self.squash.apply(_wi)


class ResSkipNet(nn.Module):
    """
    ResNet-101 encoder + multistage residual heatmap decoder
    """
    def __init__(self, cls=1, refine_encoder=False):
        super(ResSkipNet, self).__init__()
        self.cls = cls
        # squeezenet feature extractor
        self.resnet = models.resnet101(pretrained=True)
        if not refine_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # convolutions to label space
        self.heat_1 = nn.Conv2d(64, cls, 1)
        self.heat_2 = nn.Conv2d(64, cls, 1)
        self.heat_3 = nn.Conv2d(512, cls, 1)
        self.heat_4 = nn.Conv2d(1024, cls, 1)
        self.heat_5 = nn.Conv2d(2048, cls, 1)

        self.dropout = torch.nn.Dropout2d(0.1)

        # upsampling of label space heat maps
        # upsamples map_5 to size of map_4
        self.upsample_5 = nn.ConvTranspose2d(cls, cls, 3, padding=1, stride=2)
        # upsamples map_4 to size of map_3
        self.upsample_4 = nn.ConvTranspose2d(cls, cls, 3, padding=1, stride=2)
        # upsamples map_3 to size of map_2
        self.upsample_3 = nn.ConvTranspose2d(cls, cls, 3, padding=1, stride=2)
        # upsamples map_2 to size of map_1
        self.upsample_2 = nn.ConvTranspose2d(cls, cls, 3, padding=1, stride=2)
        # upsamples map_1 to original output size
        self.upsample_1 = nn.ConvTranspose2d(cls, cls, 3, padding=1, stride=2)
        self.nonlin = nn.Sigmoid()
        self.init_weights()

    def forward(self, inputs):
        siz = inputs.size()
        # reduction factor 2
        map_1 = self.resnet.conv1(inputs)
        x = self.resnet.bn1(map_1)
        x = self.resnet.relu(x)
        # reduction factor 4
        map_2 = self.resnet.maxpool(x)
        x = self.resnet.layer1(map_2)
        # reduction factor 8
        map_3 = self.resnet.layer2(x)
        # reduction factor 16
        map_4 = self.resnet.layer3(map_3)
        map_5 = self.resnet.layer4(map_4)

        map_1 = F.relu(self.dropout(self.heat_1(self.dropout(map_1))))
        map_2 = F.relu(self.dropout(self.heat_2(self.dropout(map_2))))
        map_3 = F.relu(self.dropout(self.heat_3(self.dropout(map_3))))
        map_4 = F.relu(self.dropout(self.heat_4(self.dropout(map_4))))
        map_5 = F.relu(self.dropout(self.heat_5(self.dropout(map_5))))

        # upsample using heat maps
        map_4 = map_4 + self.dropout(self.upsample_5(map_5, output_size=map_4.shape))
        map_3 = map_3 + self.dropout(self.upsample_4(map_4, output_size=map_3.shape))
        map_2 = map_2 + self.dropout(self.upsample_3(map_3, output_size=map_2.shape))
        map_1 = map_1 + self.dropout(self.upsample_2(map_2, output_size=map_1.shape))
        return self.nonlin(self.upsample_1(self.dropout(map_1), output_size=(siz[0], self.cls, siz[2], siz[3])))

    def init_weights(self):
        self.heat_1.apply(_wi)
        self.heat_2.apply(_wi)
        self.heat_3.apply(_wi)
        self.heat_4.apply(_wi)
        self.heat_5.apply(_wi)

        self.upsample_5.apply(_wi)
        self.upsample_4.apply(_wi)
        self.upsample_3.apply(_wi)
        self.upsample_2.apply(_wi)
        self.upsample_1.apply(_wi)


class DilationNet(nn.Module):
    """
    Network expanding the baseline pixel labelling to the whole line.
    """
    def __init__(self):
        super(DilationNet, self).__init__()
        #self.expand = nn.Sequential(ReNet(4, 16), nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.expand = nn.Sequential(nn.Conv2d(4, 128, 3, padding=1), nn.Conv2d(128, 32, 3, padding=1), nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.expand.apply(_wi)

    def forward(self, inputs):
        return self.expand(inputs)

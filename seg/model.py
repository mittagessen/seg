import torch

from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

from torch import nn

from PIL import Image

# NCHW
t = transforms.Compose([transforms.Resize(1200), transforms.Lambda(lambda x: x.convert('RGB')), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


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
        resnet = models.resnet50(pretrained=True)
        if not refine_encoder:
            for param in resnet.parameters():
                param.requires_grad = False
        self.net = nn.Sequential(resnet.conv1,
                                 resnet.layer1,
                                 ReNet(256, 32),
                                 ReNet(64, 32),
                                 nn.Conv2d(64, cls, 1))

        self.upsample = nn.ConvTranspose2d(cls, cls, 3, padding=1, stride=2)
        self.init_weights()

    def forward(self, inputs):
        siz = inputs.size()
        o = self.net(inputs)
        o = self.upsample(o, output_size=(siz[0], self.cls, siz[2], siz[3]))
        return o

    def init_weights(self):
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
                for p in m.parameters():
                    torch.nn.init.uniform_(p.data, -0.1, 0.1)
        self.net[2:].apply(_wi)
        self.upsample.apply(_wi)

class ResSkipNet(nn.Module):
    """
    ResNet-152 encoder + SkipNet decoder
    """
    def __init__(self, cls=4, refine_encoder=False):
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

        map_1 = F.relu(self.heat_1(map_1))
        map_2 = F.relu(self.heat_2(map_2))
        map_3 = F.relu(self.heat_3(map_3))
        map_4 = F.relu(self.heat_4(map_4))
        map_5 = F.relu(self.heat_5(map_5))

        # upsample using heat maps
        map_4 = map_4 + self.upsample_5(map_5, output_size=map_4.shape)
        map_3 = map_3 + self.upsample_4(map_4, output_size=map_3.shape)
        map_2 = map_2 + self.upsample_3(map_3, output_size=map_2.shape)
        map_1 = map_1 + self.upsample_2(map_2, output_size=map_1.shape)
        return self.upsample_1(map_1, output_size=(siz[0], self.cls, siz[2], siz[3]))

    def init_weights(self):
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
                for p in m.parameters():
                    torch.nn.init.uniform_(p.data, -0.1, 0.1)
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

class SqueezeSkipNet(nn.Module):
    """
    SqueezeNet encoder + SkipNet decoder
    """
    def __init__(self, cls=4, refine_encoder=False):
        super(SqueezeSkipNet, self).__init__()
        self.cls = cls
        # squeezenet feature extractor
        squeeze = models.squeezenet1_1(pretrained=True)
        self.feat = squeeze.features
        if not refine_encoder:
            for param in self.feat.parameters():
                param.requires_grad = False
        # convolutions to label space
        self.heat_1 = nn.Conv2d(128, cls, 1)
        self.heat_1_bn = nn.BatchNorm2d(cls)
        self.heat_2 = nn.Conv2d(256, cls, 1)
        self.heat_2_bn = nn.BatchNorm2d(cls)
        self.heat_3 = nn.Conv2d(512, cls, 1)
        self.heat_3_bn = nn.BatchNorm2d(cls)
        # upsampling of label space heat maps
        # upsamples [:]
        self.upsample_3 = nn.ConvTranspose2d(cls, cls, 2, stride=2)
        self.upsample_3_bn = nn.BatchNorm2d(cls)
        # upsamples [:7] + prev maps
        self.upsample_2 = nn.ConvTranspose2d(cls, cls, 2, stride=2)
        self.upsample_2_bn = nn.BatchNorm2d(cls)
        # upsamples [:5] + prev maps
        self.upsample_1 = nn.ConvTranspose2d(cls, cls, 5, stride=4)
        self.init_weights()

    def forward(self, inputs):
        siz = inputs.size()
        # reduction factor 4
        map_1 = self.feat[:5](inputs)
        # reduction factor 8
        map_2 = self.feat[5:7](map_1)
        # reduction factor 16
        map_3 = self.feat[7:](map_2)

        map_1 = F.relu(self.heat_1(map_1))
        map_2 = F.relu(self.heat_2(map_2))
        map_3 = F.relu(self.heat_3(map_3))

        # upsample using heat maps
        map_2 = map_2 + self.upsample_3(map_3, output_size=map_2.shape)
        map_1 = map_1 + self.upsample_2(map_2, output_size=map_1.shape)
        return self.upsample_1(map_1, output_size=(siz[0], self.cls, siz[2], siz[3]))

    def init_weights(self):
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
                for p in m.parameters():
                    torch.nn.init.uniform_(p.data, -0.1, 0.1)
        self.heat_1.apply(_wi)
        self.heat_2.apply(_wi)
        self.heat_3.apply(_wi)

        self.upsample_3.apply(_wi)
        self.upsample_2.apply(_wi)
        self.upsample_1.apply(_wi)

class ExpansionNet(nn.Module):
    """
    Network expanding the baseline pixel labelling to the whole line (shares
    feature extraction layers with BaselineNet).
    """
    def __init__(self):
        super(Expansion, self).__init__()
        self.expand = nn.Sequential(ReNet(128, 16), nn.Conv2D(32, 1, 1), nn.Sigmoid())

    def forward(self, baselines, features):
        o = self.expand(inputs)
        o = F.upsample(o, inputs.shape[1:], mode='bilinear')
        return o

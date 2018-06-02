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


class BaselineNet(nn.Module):
    """
    Baseline labelling network.
    """
    def __init__(self):
        super(BaselineNet, self).__init__()
        squeeze = models.squeezenet1_1(pretrained=True)
        self.feat = squeeze.features[:5]
        for param in self.feat.parameters():
            param.requires_grad = False
        self.label = nn.Sequential(ReNet(128, 32), nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.init_weights()

    def forward(self, inputs):
        features = self.feat(inputs)
        o = self.label(features)
        o = F.upsample(o, inputs.shape[2:], mode='bilinear')
        return o, features

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
        self.label.apply(_wi)


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

#!/usr/bin/env python3
import numpy as np

import torchvision.transforms.functional as tf

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

from seg import model
from seg.dataset import BaselineSet

from torchvision import transforms
from PIL import Image
import click

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

class EarlyStopping(object):
    """
    Early stopping to terminate training when validation loss doesn't improve
    over a certain time.
    """
    def __init__(self, it=None, min_delta=0.002, lag=5):
        """
        Args:
            it (torch.utils.data.DataLoader): training data loader
            min_delta (float): minimum change in validation loss to qualify as improvement.
            lag (int): Number of epochs to wait for improvement before
                       terminating.
        """
        self.min_delta = min_delta
        self.lag = lag
        self.it = it
        self.best_loss = 0
        self.wait = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.wait >= self.lag:
             raise StopIteration
        return self.it

    def update(self, val_loss):
        """
        Updates the internal validation loss state
        """
        if (val_loss - self.best_loss) < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            self.best_loss = val_loss

@click.group()
def cli():
    pass

@cli.command()
@click.option('-i', '--load', type=click.Path(exists=True, readable=True), help='pretrained weights to load')
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-t', '--arch', default='ResUNet', type=click.Choice(['ResUNet', 'ResSkipNet']))
@click.option('-l', '--lrate', default=0.03, help='initial learning rate')
@click.option('--weight-decay', default=0.0, help='weight decay')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.option('--refine-projection/--clear-projection', default=False, help='clear pretrained last layer')
@click.option('-r', '--refine-encoder/--freeze-encoder', default=False, help='freeze pretrained encoder weights')
@click.option('--lag', show_default=True, default=20, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('--augment/--no-augment', show_default=True, default=True, help='Enables/disables data augmentation')
@click.option('--weigh-loss/--no-weigh-loss', show_default=True, default=True, help='Weighs cross entropy loss for class frequency')
@click.option('--optimizer', show_default=True, default='SGD', type=click.Choice(['SGD', 'Adam']), help='optimizer')
@click.option('--crf/--no-crf', show_default=True, default=True, help='enables CRF postprocessing')
@click.option('--threads', default=min(len(os.sched_getaffinity(0)), 4))
@click.argument('ground_truth', nargs=1)
def train(load, name, arch, lrate, weight_decay, workers, device, validation,
          refine_projection, refine_encoder, lag, min_delta, augment, weigh_loss, optimizer, crf,
          threads, ground_truth):

    print('model output name: {}'.format(name))

    torch.set_num_threads(threads)

    train_set = BaselineSet(glob.glob('{}/**/*.tif'.format(ground_truth), recursive=True), augment=augment)
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_set = BaselineSet(glob.glob('{}/**/*.tif'.format(validation), recursive=True), augment=False)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    device = torch.device(device)

    print('loading network')
    net = getattr(model, arch)(4, refine_encoder).to(device, non_blocking=True)
    if load:
        print('loading weights')
        net.load_state_dict(torch.load(load, map_location='cpu'))
        if not refine_projection:
            print('reinitialiizing last layer')
            model._wi(net.squash)

    weights = None
    if weigh_loss:
        print('calculating class proportions')
        weights = train_set.get_target_weights()
        print(weights)
        weights = weights.to(device, non_blocking=True)
    criterion = nn.CrossEntropyLoss(weights)

    if optimizer == 'SGD':
        opti = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lrate, weight_decay=weight_decay)
    else:
        opti = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lrate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(opti, patience=5, mode='max', verbose=True)
    st_it = EarlyStopping(train_data_loader, min_delta, lag)

    for epoch, loader in enumerate(st_it):
        epoch_loss = 0
        with click.progressbar(train_data_loader, label='epoch {}'.format(epoch)) as bar:
            for sample in bar:
                input, target = sample[0].to(device, non_blocking=True), sample[1].to(device, non_blocking=True)
                opti.zero_grad()
                o = net(input)
                loss = criterion(o, target)
                epoch_loss += loss.item()
                loss.backward()
                opti.step()
        torch.save(net.state_dict(), '{}_{}.ckpt'.format(name, epoch))
        print("===> epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        val_loss, thresh_loss, crf_loss = evaluate(net, device, val_data_loader)
        net.train()
        if optimizer == 'SGD':
            scheduler.step(val_loss)
        st_it.update(val_loss)
        print("===> epoch {} validation loss: {:.4f} (thresholded: {:.4f}, crf: {:.4f})".format(epoch, val_loss, thresh_loss, crf_loss))

def evaluate(model, device, data_loader, threshold=0.5):
   """
   Calculates argmax and softmax > 0.5 thresholded accuracy, + accuracy after
   CRF postprocessing.
   """
   model.eval()
   aaccuracy = 0.0
   taccuracy = 0.0
   caccuracy = 0.0
   with torch.no_grad():
        for sample in data_loader:
            input, target, res = sample[0].to(device, non_blocking=True), sample[1].to(device, non_blocking=True), tf.to_pil_image(sample[2].squeeze())
            o = model(input)
            # argmax accuracy
            pred = torch.argmax(o, 1).squeeze()
            tp = float(pred.eq(target).sum())
            aaccuracy += tp / len(target.view(-1))
            # thresholded accuracy
            probs = F.softmax(o, dim=1).squeeze()
            pred = torch.argmax(probs > 0.5, dim=0)
            tp = float(pred.eq(target).sum())
            taccuracy += tp / len(target.view(-1))
            # crf accuracy
            pred = run_crf(res, probs)
            tp = float(pred.eq(target.cpu().squeeze()).sum())
            caccuracy += tp / len(target.view(-1))
   return aaccuracy / len(data_loader), taccuracy / len(data_loader), caccuracy / len(data_loader)

def run_crf(img, output):
    d = dcrf.DenseCRF2D(img.size[0], img.size[1], output.size(0))
    # unary energy
    # 4 x H x W
    u = unary_from_softmax(output.cpu().detach().numpy())
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(49, 49), srgb=(5, 5, 5), rgbim=np.array(img), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    q = d.inference(5)
    return torch.tensor(np.argmax(q, axis=0)).reshape(*img.size[::-1])

@cli.command()
@click.option('-m', '--model', default=None, help='model file')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.argument('images', nargs=-1)
def pred(model, device, images):
    m = ConvReNet(4)
    m.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
    device = torch.device(device)
    m.to(device, non_blocking=True)

    resize = transforms.Resize(1200)
    transform = transforms.Compose([transforms.Resize(1200), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    cmap = {0: (230, 25, 75, 127),
            1: (60, 180, 75, 127),
            2: (255, 225, 25, 127),
            3: (0, 130, 200, 127)}

    with torch.no_grad():
        for img in images:
            print('transforming image {}'.format(img))
            im = Image.open(img).convert('RGB')
            norm_im = transform(im)
            print('running forward pass')
            o = m.forward(norm_im.unsqueeze(0))
            probs = F.softmax(o, dim=1).squeeze()
            print('CRF postprocessing')
            o = run_crf(resize(im), probs)
            print('result extraction')
            # resample to original size
            cls = np.array(Image.fromarray(np.array(o, 'uint8')).resize(im.size, resample=Image.NEAREST))
            overlay = np.zeros(im.size[::-1] + (4,))

            for idx, val in cmap.items():
                overlay[cls == idx] = val
                layer = np.full(im.size[::-1], 255)
                layer[cls == idx] = 0
                Image.fromarray(layer.astype('uint8')).resize(im.size).save(os.path.splitext(img)[0] + '_class_{}.png'.format(idx))
            print('saving output(s)')
            im = Image.alpha_composite(im.convert('RGBA'), Image.fromarray(overlay.astype('uint8'))).resize(im.size)
            im.save(os.path.splitext(img)[0] + '_overlay.png')


if __name__ == '__main__':
    cli()


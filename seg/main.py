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

from seg.model import ConvReNet, ResSkipNet, ResUNet, DilationNet
from seg.dataset import BaselineSet, DilationSet, dilate_collate

from scipy.misc import imsave
from torchvision import transforms
from PIL import Image
import click

from scipy.ndimage import label

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
        #if self.wait >= self.lag:
        #     raise StopIteration
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
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names', show_default=True)
@click.option('-l', '--lrate', default=0.3, help='initial learning rate', show_default=True)
@click.option('-w', '--workers', default=0, help='number of workers loading training data', show_default=True)
@click.option('-d', '--device', default='cpu', help='pytorch device', show_default=True)
@click.option('-b', '--batch-size', default=8, help='batch size', show_default=True)
@click.option('-v', '--validation', default='val', help='validation set location', show_default=True)
@click.option('--lag', default=20, help='Number of epochs to wait before stopping training without improvement', show_default=True)
@click.option('--min-delta', default=0.005, help='Minimum improvement between epochs to reset early stopping', show_default=True)
@click.option('--optimizer', default='Adam', type=click.Choice(['SGD', 'Adam']), help='optimizer', show_default=True)
@click.option('--threads', default=min(len(os.sched_getaffinity(0)), 4), show_default=True)
@click.argument('ground_truth', nargs=1)
def train_dilation(name, lrate, workers, device, batch_size, validation, lag, min_delta, optimizer, threads, ground_truth):

    print('model output name: {}'.format(name))

    torch.set_num_threads(threads)

    train_set = DilationSet(glob.glob('{}/**/*.seeds.png'.format(ground_truth), recursive=True), augment=False)
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, collate_fn=dilate_collate, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_set = DilationSet(glob.glob('{}/**/*.seeds.png'.format(validation), recursive=True), augment=False)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, collate_fn=dilate_collate, batch_size=batch_size, pin_memory=True)

    device = torch.device(device)

    print('loading network')
    model = DilationNet().to(device)
    criterion = nn.BCELoss()

    if optimizer == 'SGD':
        opti = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    else:
        opti = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    scheduler = lr_scheduler.ReduceLROnPlateau(opti, patience=5, mode='max', verbose=True)
    st_it = EarlyStopping(train_data_loader, min_delta, lag)
    val_loss = 1.0
    for epoch, loader in enumerate(st_it):
        epoch_loss = 0
        with click.progressbar(train_data_loader, label='epoch {}'.format(epoch), show_pos=True) as bar:
            for sample in bar:
                lens, input, target = sample[0].to(device, non_blocking=True),\
                                      sample[1].to(device, non_blocking=True),\
                                      sample[2].to(device, non_blocking=True)
                opti.zero_grad()
                o, olens = model(input, lens)
                # flatten tensor and remove masked bits
                mask = torch.zeros_like(o)
                for idx, x in enumerate(olens.int()):
                    mask[idx, :, :, 0:x] = torch.ones((mask.shape[1], mask.shape[2], x))
                print(o[mask].shape)
                loss = criterion(o, target)
                epoch_loss += loss.item()
                loss.backward()
                opti.step()
        torch.save(model.state_dict(), '{}_{}.ckpt'.format(name, epoch))
        print("===> epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        val_acc, val_recall, val_precision, val_loss = evaluate(model, device, criterion, val_data_loader)
        model.train()
        #if optimizer == 'SGD':
        #    scheduler.step(val_loss)
        ##st_it.update(val_loss)
        imsave('{:06d}.png'.format(epoch), o.detach().cpu().squeeze().numpy())
        print("===> epoch {} validation loss: {:.4f} (accuracy: {:.4f}, recall: {:.4f}, precision: {:.4f})".format(epoch, val_loss, val_acc, val_recall, val_precision))

@cli.command()
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-t', '--arch', default='ResUNet', type=click.Choice(['ResSkipNet', 'ConvReNet', 'ResUNet']))
@click.option('-l', '--lrate', default=0.003, help='initial learning rate')
@click.option('--weight-decay', default=1e-5, help='weight decay')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.option('-r', '--refine-encoder/--freeze-encoder', default=False, help='Freeze pretrained encoder weights')
@click.option('--lag', show_default=True, default=20, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('--optimizer', show_default=True, default='SGD', type=click.Choice(['SGD', 'Adam']), help='optimizer')
@click.option('--threads', default=min(len(os.sched_getaffinity(0)), 4))
@click.option('--weigh-loss/--no-weigh-loss', show_default=True, default=True, help='Weighs cross entropy loss for class frequency')
@click.option('--augment/--no-augment', show_default=True, default=True, help='Enables data augmentation')
@click.argument('ground_truth', nargs=1)
def train(name, arch, lrate, weight_decay, workers, device, validation, refine_encoder, lag,
          min_delta, optimizer, threads, weigh_loss, augment,
          ground_truth):

    print('model output name: {}'.format(name))

    torch.set_num_threads(threads)

    train_set = BaselineSet(glob.glob('{}/**/*.seeds.png'.format(ground_truth), recursive=True), augment=augment)
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_set = BaselineSet(glob.glob('{}/**/*.seeds.png'.format(validation), recursive=True), augment=False)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    device = torch.device(device)

    print('loading network')
    if arch == 'ResSkipNet':
        model = ResSkipNet(1, refine_encoder).to(device)
    elif arch == 'ConvReNet':
        model = ConvReNet(1, refine_encoder).to(device)
    elif arch == 'ResUNet':
        model = ResUNet(1, refine_encoder).to(device)
    else:
        raise Exception('invalid model type selected')

    weights = None
    if weigh_loss:
        weights = train_set.get_target_weights()
        print(weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

    if optimizer == 'SGD':
        opti = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
    else:
        opti = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(opti, patience=5, mode='max', verbose=True)
    st_it = EarlyStopping(train_data_loader, min_delta, lag)
    val_loss = 1.0
    for epoch, loader in enumerate(st_it):
        epoch_loss = 0
        with click.progressbar(train_data_loader, label='epoch {}'.format(epoch)) as bar:
            for sample in bar:
                input, target = sample[0].to(device, non_blocking=True), sample[1].to(device, non_blocking=True)
                opti.zero_grad()
                o = model(input)
                loss = criterion(o, target)
                epoch_loss += loss.item()
                loss.backward()
                opti.step()
        torch.save(model.state_dict(), '{}_{}.ckpt'.format(name, epoch))
        print("===> epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        val_acc, val_loss = evaluate(model, device, criterion, val_data_loader)
        model.train()
        if optimizer == 'SGD':
            scheduler.step(val_loss)
        #st_it.update(val_loss)
        imsave('{:06d}.png'.format(epoch), o.detach().cpu().squeeze().numpy())
        print("===> epoch {} validation loss: {:.4f} (accuracy: {:.4f})".format(epoch, val_loss, val_acc))


def hysteresis_thresh(im, low, high):
    lower = im > low
    components, count = label(lower, np.ones((3, 3)))
    valid = np.unique(components[lower & (im > high)])
    lm = np.zeros((count + 1,), bool)
    lm[valid] = True
    return lm[components]

def evaluate(model, device, criterion, data_loader):
    """
    """
    model.eval()
    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    loss = 0.0
    with torch.no_grad():
         for sample in data_loader:
             input, target = sample[0].to(device), sample[1].to(device)
             o = model(input)
             loss += criterion(o, target)
             o = model.nonlin(o)
             pred = hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.3, 0.5)
             tp = float((pred == target.detach().squeeze().cpu().numpy()).sum())
             recall += tp / target.sum()
             precision += tp / pred.sum()
             accuracy += tp / len(target.view(-1))
    return accuracy / len(data_loader), recall / len(data_loader), precision / len(data_loader), loss / len(data_loader)

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
    m = ConvReNet(1)
    m.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
    device = torch.device(device)
    m.to(device)

    resize = transforms.Resize(1200)
    transform = transforms.Compose([transforms.Resize(1200), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    with torch.no_grad():
        for img in images:
            print('transforming image {}'.format(img))
            im = Image.open(img).convert('RGB')
            norm_im = transform(im)
            print('running forward pass')
            o = m.forward(norm_im.unsqueeze(0))
            cls = Image.fromarray((o.detach().squeeze().cpu().numpy()*255).astype('uint8')).resize(im.size, resample=Image.NEAREST)
            cls.save(os.path.splitext(img)[0] + '_nonthresh.png')
            o = hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.3, 0.5)
            print('result extraction')
            # resample to original size
            cls = Image.fromarray(np.array(o, 'uint8')).resize(im.size, resample=Image.NEAREST)
            cls.save(os.path.splitext(img)[0] + '_class.png')


if __name__ == '__main__':
    cli()


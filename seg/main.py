#!/usr/bin/env python3
import numpy as np

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model import ConvReNet, SqueezeSkipNet
from dataset import BaselineSet

from scipy.misc import imshow, imsave

import click

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
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-t', '--arch', default='SqueezeSkipNet', type=click.Choice(['SqueezeSkipNet', 'ConvReNet']))
@click.option('-l', '--lrate', default=0.03, help='initial learning rate')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.option('-w', '--refine-encoder/--freeze-encoder', default=False, help='Freeze pretrained encoder weights')
@click.option('--lag', show_default=True, default=5, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('--threads', default=min(len(os.sched_getaffinity(0)), 4))
@click.argument('ground_truth', nargs=1)
def train(name, arch, lrate, workers, device, validation, threads, refine_encoder, lag, min_delta, ground_truth):

    torch.set_num_threads(threads)

    train_set = BaselineSet(glob.glob('{}/**/*.jpg'.format(ground_truth), recursive=True))
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_set = BaselineSet(glob.glob('{}/**/*.jpg'.format(validation), recursive=True))
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    device = torch.device(device)

    if arch == 'SqueezeSkipNet':
        model = SqueezeSkipNet(4, refine_encoder).to(device)
    elif arch == 'ConvReNet':
        model = ConvReNet(4, refine_encoder).to(device)
    else:
        raise click
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    st_it = EarlyStopping(train_data_loader, min_delta, lag)

    for epoch, loader in enumerate(st_it):
        epoch_loss = 0
        with click.progressbar(train_data_loader, label='epoch {}'.format(epoch)) as bar:
            for sample in bar:
                input, target = sample[0].to(device), sample[1].to(device)
                optimizer.zero_grad()
                o = model(input)
                loss = criterion(o, target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), '{}_{}.ckpt'.format(name, epoch))
        print("===> epoch {} complete: avg. loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        val_loss = evaluate(model, criterion, device, val_data_loader)
        model.train()
        scheduler.step(val_loss)
        st_it.update(val_loss)
        print("===> epoch {} validation loss: {:.4f}".format(epoch, val_loss))
        #imsave('epoch_{}.png'.format(epoch), o.detach().squeeze().numpy())

def evaluate(model, criterion, device, data_loader):
   model.eval()
   val_loss = 0.0
   with torch.no_grad():
        for sample in data_loader:
            input, target = sample[0].to(device), sample[1].to(device)
            o = model(input)
            val_loss += float(criterion(o, target))
   return val_loss / len(data_loader)


@cli.command()
def pred():
    pass

if __name__ == '__main__':
    cli()


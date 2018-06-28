#!/usr/bin/env python3
import numpy as np

import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model import BaselineNet, SqueezeSkipNet
from dataset import BaselineSet

from scipy.misc import imshow, imsave

import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-b', '--batch-size', default=32, help='batch size')
@click.option('-e', '--epochs', default=100, help='training time')
@click.option('-l', '--lrate', default=0.01, help='learning rate')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.argument('ground_truth', nargs=1)
def train(name, batch_size, epochs, lrate, workers, device, validation, ground_truth):

    train_set = BaselineSet(glob.glob('{}/**/*.jpg'.format(ground_truth), recursive=True))
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True)
    val_set = BaselineSet(glob.glob('{}/**/*.jpg'.format(validation), recursive=True))
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1)

    device = torch.device(device)

    model = SqueezeSkipNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    for epoch in range(epochs):
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
        print("===> epoch {} validation loss: {:.4f}".format(epoch, evaluate(model, criterion, device, val_data_loader)))
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


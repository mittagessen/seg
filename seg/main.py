#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model import BaselineNet
from dataset import BaselineSet

from scipy.misc import imshow, imsave

import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('-e', '--epochs', default=100, help='training time')
@click.option('-l', '--lrate', default=0.01, help='learning rate')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-p', '--partition', default=0.95, help='default validation set partition')
@click.argument('ground_truth', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def train(epochs, lrate, workers, device, partition, ground_truth):
    ground_truth = list(ground_truth)
    np.random.shuffle(ground_truth)
    train_im = ground_truth[:int(len(ground_truth) * partition)]
    val_im = ground_truth[int(len(ground_truth) * partition):]
    train_set = BaselineSet(train_im)
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True)
    val_set = BaselineSet(val_im)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1)

    device = torch.device(device)

    model = BaselineNet().to(device)
    criterion = nn.MSELoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
    for epoch in range(epochs):
        epoch_loss = 0
        for idx, sample in enumerate(train_data_loader, 1):
            input, target = sample[0].to(device), sample[1].to(device)
            optimizer.zero_grad()
            o, _ = model(input)
            loss = criterion(o, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, idx, len(train_data_loader), loss.item()))
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_data_loader)))
        print("===> Epoch {} validation loss: {:.4f}".format(epoch, evaluate(model, criterion, device, val_data_loader)))
        imsave('epoch_{}.png'.format(epoch), o.detach().squeeze().numpy())
    pass

def evaluate(model, criterion, device, data_loader):
   model.eval()
   val_loss = 0.0
   with torch.no_grad():
        for sample in data_loader:
            input, target = sample[0].to(device), sample[1].to(device)
            o, _ = model(input)
            val_loss += float(criterion(o, target))
   return val_loss / len(data_loader)


@cli.command()
def pred():
    pass

if __name__ == '__main__':
    cli()


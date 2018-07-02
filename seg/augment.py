#!/usr/bin/env python3
import numpy as np
import Augmentor
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

@click.group()
def cli():
    pass

@cli.command()
@click.option('-i', '--images', default='imgs', help='directory with input images')
@click.option('-l', '--labels', default='labels', help='directory with labelled images for inputs')
@click.option('-s', '--samples', default=1000, help='number of augmented output images to generate')
@click.option('-o', '--output', default='output', help='output directory')
def augment(images, labels, samples, output):
    p = Augmentor.Pipeline(images, output_directory=output)
    p.ground_truth(labels)
    p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
    p.zoom_random(probability=0.5, percentage_area=0.9, randomise_percentage_area=True)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(samples, multi_threaded=False)


if __name__ == '__main__':
    cli()


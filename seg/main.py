#!/usr/bin/env python3
import numpy as np

import torchvision.transforms.functional as tf

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from seg.model import ResUNet
from seg.dataset import BaselineSet
from seg.postprocess import denoising_hysteresis_thresh, vectorize_lines

from scipy.misc import imsave
from torchvision import transforms
from PIL import Image
import click

from scipy.ndimage import label


from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage

@click.group()
def cli():
    pass


@cli.command()
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-t', '--arch', default='ResUNet', type=click.Choice(['ResUNet']))
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
@click.option('--weigh-loss/--no-weigh-loss', show_default=True, default=False, help='Weighs cross entropy loss for class frequency')
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
    if arch == 'ResUNet':
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

    def score_function(engine):
        val_loss = engine.state.metrics['accuracy']
        return -val_loss

    def output_preprocess(output):
        o, target = output
        o = torch.sigmoid(o)
        o = denoising_hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.3, 0.5, 2.5)
        return torch.from_numpy(o.astype('f')).unsqueeze(0).unsqueeze(0), target.double()

    trainer = create_supervised_trainer(model, opti, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, device=device, non_blocking=True, metrics={'accuracy': Accuracy(output_transform=output_preprocess),
                                                                                              'precision': Precision(output_transform=output_preprocess),
                                                                                              'recall': Recall(output_transform=output_preprocess)})
    ckpt_handler = ModelCheckpoint('.', name, save_interval=1, n_saved=10, require_empty=False)
    est_handler = EarlyStopping(lag, score_function, traine)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, ['loss'])

    evaluator.add_event_handler(Events.COMPLETED, est_handler)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=ckpt_handler, to_save={'net': model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        progress_bar.log_message('eval results - epoch {} accuracy: {:.2f} recall: {:.2f} precision {:.2f}'.format(engine.state.epoch,
                                                                                                                   metrics['accuracy'],
                                                                                                                   metrics['recall'],
                                                                                                                   metrics['precision']))
    trainer.run(train_data_loader)


@cli.command()
@click.option('-m', '--model', default=None, help='model file')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.argument('images', nargs=-1)
def pred(model, device, images):

    m = ResUNet(1)
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


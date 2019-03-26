#!/usr/bin/env python3
import numpy as np

import torchvision.transforms.functional as tf

import os
import glob
import json
import click
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image, ImageDraw
from scipy.misc import imsave
from torchvision import transforms
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.contrib.handlers import ProgressBar

from seg.model import ResUNet, _wi
from seg.dataset import BaselineSet
from seg.postprocess import denoising_hysteresis_thresh, vectorize_lines, line_extractor


@click.group()
def cli():
    pass


@cli.command()
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-i', '--load', default=None, type=click.Path(exists=True, readable=True), help='pretrained weights to load')
@click.option('-t', '--arch', default='ResUNet', type=click.Choice(['ResUNet']))
@click.option('-l', '--lrate', default=0.003, help='initial learning rate')
@click.option('--weight-decay', default=1e-5, help='weight decay')
@click.option('-w', '--workers', default=0, help='number of workers loading training data')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-v', '--validation', default='val', help='validation set location')
@click.option('--refine-projection/--clear-projection', default=False, help='clear pretrained last layer')
@click.option('-r', '--refine-encoder/--freeze-encoder', default=False, help='Freeze pretrained encoder weights')
@click.option('--lag', show_default=True, default=20, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('--optimizer', show_default=True, default='SGD', type=click.Choice(['SGD', 'Adam']), help='optimizer')
@click.option('--threads', default=min(len(os.sched_getaffinity(0)), 4))
@click.option('--weigh-loss/--no-weigh-loss', show_default=True, default=False, help='Weighs cross entropy loss for class frequency')
@click.option('--augment/--no-augment', show_default=True, default=True, help='Enables data augmentation')
@click.argument('ground_truth', nargs=1)
def train(name, load, arch, lrate, weight_decay, workers, device, validation,
          refine_projection, refine_encoder, lag, min_delta, optimizer,
          threads, weigh_loss, augment, ground_truth):

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

    if load:
        print('loading weights')
        model = torch.load(load, map_location='cpu')
        if not refine_projection:
            print('reinitializing last layer')
            _wi(model.squash)

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
        val_loss = engine.state.metrics['loss']
        return -val_loss

    def output_preprocess(output):
        o, target = output
        o = torch.sigmoid(o)
        o = denoising_hysteresis_thresh(o.detach().squeeze().cpu().numpy(), 0.3, 0.5, 2.5)
        return torch.from_numpy(o.astype('f')).unsqueeze(0).unsqueeze(0).to(device), target.double().to(device)

    trainer = create_supervised_trainer(model, opti, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, device=device, non_blocking=True, metrics={'accuracy': Accuracy(output_transform=output_preprocess),
                                                                                              'precision': Precision(output_transform=output_preprocess),
                                                                                              'recall': Recall(output_transform=output_preprocess),
                                                                                              'loss': Loss(criterion)})
    ckpt_handler = ModelCheckpoint('.', name, save_interval=1, n_saved=10, require_empty=False)
    est_handler = EarlyStopping(lag, score_function, trainer)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, ['loss'])

    evaluator.add_event_handler(Events.COMPLETED, est_handler)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=ckpt_handler, to_save={'net': model})
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=TerminateOnNan())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        progress_bar.log_message('eval results - epoch {} loss: {:.2f} accuracy: {:.2f} recall: {:.2f} precision {:.2f}'.format(engine.state.epoch,
                                                                                                                   metrics['loss'],
                                                                                                                   metrics['accuracy'],
                                                                                                                   metrics['recall'],
                                                                                                                   metrics['precision']))
    trainer.run(train_data_loader, max_epochs=1000)


@cli.command()
@click.option('-m', '--model', default=None, help='model file')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-c', '--context', default=80, help='context around baseline')
@click.option('-t', '--thresholds', default=(0.3, 0.5), type=(float, float), help='thresholds for hysteresis thresholding')
@click.option('-s', '--sigma', default=2.5, help='sigma of gaussian filter in postprocessing')
@click.argument('images', nargs=-1)
def pred(model, device, context, thresholds, sigma, images):

    device = torch.device(device)
    with open(model, 'rb') as fp:
        m = torch.load(fp, map_location=device)

    resize = transforms.Resize(1200)
    transform = transforms.Compose([transforms.Resize(1200), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    with torch.no_grad():
        for img in images:
            print('transforming image {}'.format(img))
            im = Image.open(img).convert('RGB')
            norm_im = transform(im).to(device)
            print('running forward pass')
            o = m.forward(norm_im.unsqueeze(0))
            o = torch.sigmoid(o)
            cls = Image.fromarray((o.detach().squeeze().cpu().numpy()*255).astype('uint8')).resize(im.size, resample=Image.NEAREST)
            cls.save(os.path.splitext(img)[0] + '_nonthresh.png')
            o = denoising_hysteresis_thresh(o.detach().squeeze().cpu().numpy(), thresholds[0], thresholds[1], sigma)
            cls = Image.fromarray((o*255).astype('uint8')).resize(im.size, resample=Image.NEAREST)
            cls.save(os.path.splitext(img)[0] + '_thresh.png')
            print('result extraction')
            # running line vectorization
            lines = vectorize_lines(np.array(cls))
            #with open('{}.txt'.format(os.path.splitext(img)[0]), 'w') as fp:
            #    for line in lines:
            #        fp.write(';'.join(['{},{}'.format(x[0], x[1]) for x in line]) + '\n')
            #with open('{}.json'.format(os.path.splitext(img)[0]), 'w') as fp:
            #    json.dump(lines, fp)
            #for idx, line in enumerate(lines):
            #    l = line_extractor(np.array(im.convert('L')), line, 80)
            #    Image.fromarray(line_extractor(np.array(im.convert('L')), line, 80)).save('{}_{}.png'.format(os.path.splitext(img)[0], idx))
            overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for line in lines:
                draw.line([tuple(x[::-1]) for x in line], width=10, fill=(0, 130, 200, 127))
            del draw
            Image.alpha_composite(im.convert('RGBA'), overlay).save('{}_overlay.png'.format(os.path.splitext(img)[0]))

if __name__ == '__main__':
    cli()


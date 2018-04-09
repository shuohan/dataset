#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from plotting_utils import get_colors


parser = argparse.ArgumentParser(description='Show the convergence')
parser.add_argument('logs', nargs='+',
                    help='The .csv file of the training process. The first '
                         'column is epoch ID, the second column is the '
                         'training loss, and the third column is the '
                         'validation loss.')
parser.add_argument('-n', '--names', nargs='+', default=None, required=False,
                    help='Legend name of all logs')
parser.add_argument('-o', '--output', default=None, required=False,
                    help='Save fig to path')
parser.add_argument('-l', '--ylim', nargs=2, type=int, default=[0, 1],
                    required=False, help='y lim')
parser.add_argument('-c', '--colormap', default='tab20')
parser.add_argument('-m', '--show-mean', action='store_true', default=False)
args = parser.parse_args()

if args.names is None:
    args.names = args.logs

fig = plt.figure(figsize=(8,6), dpi=100)
num_colors = 20
colormap = args.colormap
colors = get_colors(num_colors, colormap)

legend = list()
validation_losses = list()
training_losses = list()
for log, color, name in zip(args.logs, colors, args.names):
    with open(log) as log_file:
        reader = csv.reader(log_file)
        next(reader)
        rows = list(reader)
        epochs, training_loss, validation_loss = zip(*rows)
    epochs = [int(e) for e in epochs]
    training_loss = [float('%.4f' % float(tl)) for tl in training_loss]
    validation_loss = [float('%.4f' % float(vl)) for vl in validation_loss]
    training_losses.append(training_loss)
    validation_losses.append(validation_loss)
    plt.plot(epochs, training_loss, color=color)
    plt.plot(epochs, validation_loss, '-.', color=color)
    legend.append(name + ' training loss')
    legend.append(name + ' validation loss')


if args.show_mean:
    validation_mean = np.mean(validation_losses, axis=0)
    training_mean = np.mean(training_losses, axis=0)
    plt.plot(epochs, training_mean, color=colors[-1], linewidth=3)
    plt.plot(epochs, validation_mean, '-.', color=colors[-1], linewidth=3)
    min_idx = np.argmin(validation_mean)
    plt.plot([epochs[min_idx], epochs[min_idx]], args.ylim, 'r')
    legend.append('mean training loss')
    legend.append('mean validation loss')
        
plt.legend(legend, loc=2, prop={'size': 6})
plt.yticks(np.arange(args.ylim[0], args.ylim[1],
                     (args.ylim[1] - args.ylim[0]) / 20))
plt.ylim(args.ylim)
plt.grid('on')
if args.output is not None:
    fig.savefig(args.output, dpi=220)
else:
    plt.show()

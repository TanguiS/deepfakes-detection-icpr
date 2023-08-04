from pathlib import Path
from typing import Tuple, List, Dict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from evaluation.util import read_models_information
from plot import util


def plot_training_results(model_dir: Path, root_event_path: Path) -> None:
    face_policy, patch_size, net_name, model_name = read_models_information(model_dir)
    shape = (patch_size, patch_size, 3)
    tb_file_path = [file for file in root_event_path.rglob(model_dir.name)][0]

    train_data, val_data = util.tensorboard_to_dict_data(tb_file_path)
    plot_train_history(train_data, net_name, shape)
    plot_val_history(val_data, net_name, shape)


def plot_train_history(train_data: Dict[str, List], arch: str, shape: Tuple[int, int, int]):
    plt.style.use('seaborn-whitegrid')
    gs = gridspec.GridSpec(2, 4, wspace=1., hspace=.35)
    fig: plt.Figure = plt.figure(figsize=(8, 8))

    train_time = [train_data['time (s)'][0][0]]
    for i in range(1, len(train_data['time (s)'][0])):
        train_time.append(train_time[-1] + train_data['time (s)'][0][i])

    ax1: plt.Axes = fig.add_subplot(gs[0, 0:4])
    ax1.plot(train_time, train_data['time (s)'][1], linewidth=3, color='orange')
    ax1.set_title('Training Time (s)', fontsize=16)
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Epoch", fontsize=16)

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(train_data['train/loss'][0], train_data['train/loss'][1], linewidth=3, color='royalblue')
    ax2.set_title('Training Loss', fontsize=16)
    ax2.set_xlabel("Epoch", fontsize=16)
    ax2.set_ylabel("Loss", fontsize=16)

    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.plot(train_data['train/roc_auc'][0], train_data['train/roc_auc'][1], linewidth=3, color='mediumseagreen')
    ax3.set_title('Training Accuracy', fontsize=16)
    ax3.set_xlabel("Epoch", fontsize=16)
    ax3.set_ylabel("Accuracy", fontsize=16)

    fig.suptitle(f'Training History - {arch} / {shape}', fontsize=24)
    plt.show()


def plot_val_history(val_data: Dict[str, List], arch: str, shape: Tuple[int, int, int]):

    plt.style.use('seaborn-whitegrid')
    gs = gridspec.GridSpec(2, 2, wspace=.35, hspace=.35)
    fig: plt.Figure = plt.figure(figsize=(8, 8))

    train_time = [val_data['time (s)'][0][0]]
    for i in range(1, len(val_data['time (s)'][0])):
        train_time.append(train_time[-1] + val_data['time (s)'][0][i])

    ax1: plt.Axes = fig.add_subplot(gs[0, 0])
    ax1.plot(train_time, val_data['time (s)'][1], linewidth=3, color='orange')
    ax1.set_title('Training Time (s)', fontsize=16)
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Epoch", fontsize=16)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(val_data['lr'][0], val_data['lr'][1], linewidth=3, color='violet')
    ax1.set_title('Learning Rate', fontsize=16)
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("Learning rate", fontsize=16)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(val_data['val/loss'][0], val_data['val/loss'][1], linewidth=3, color='royalblue')
    ax3.set_title('Validation Loss', fontsize=16)
    ax3.set_xlabel("Epoch", fontsize=16)
    ax3.set_ylabel("Loss", fontsize=16)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(val_data['val/roc_auc'][0], val_data['val/roc_auc'][1], linewidth=3, color='mediumseagreen')
    ax4.set_title('Validation Accuracy', fontsize=16)
    ax4.set_xlabel("Epoch", fontsize=16)
    ax4.set_ylabel("Accuracy", fontsize=16)

    fig.suptitle(f'Validation History - {arch} / {shape}', fontsize=24)
    plt.show()

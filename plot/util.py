from pathlib import Path
from typing import Dict, List

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tensorboard_to_dict_data(tb_file_path: Path):
    event_acc = EventAccumulator(str(tb_file_path))
    event_acc.Reload()

    tags = event_acc.Tags()['scalars']
    data = {tag: [] for tag in tags}

    for tag in tags:
        event = event_acc.Scalars(tag)
        values = [scalar.value for scalar in event]
        steps = [scalar.step for scalar in event]
        data[tag].extend([steps, values])
    event = event_acc.Scalars(tags[0])
    data['wall time'] = [[scalar.step for scalar in event], [scalar.wall_time for scalar in event]]

    val_data = get_val_data(data)
    train_data = get_train_data(data)
    del data

    return train_data, val_data


def get_train_data(data: Dict[str, List]) -> Dict[str, List]:
    train_data = data.copy()
    del data

    for tag in ['train/loss', 'train/roc_auc']:
        scale_data(tag, train_data)

    scale_time_data(train_data, 'train/loss')

    return train_data


def get_val_data(data: Dict[str, List]) -> Dict[str, List]:
    val_data = {}
    for tag in ['val/loss', 'val/roc_auc', 'lr']:
        val_data[tag] = data[tag].copy()
        del data[tag]
    val_data['wall time'] = data['wall time'].copy()
    val_data['epoch'] = data['epoch'].copy()

    for tag in ['val/loss', 'val/roc_auc', 'lr']:
        scale_data(tag, val_data)

    scale_time_data(val_data, 'lr')

    return val_data


def scale_data(tag: str, train_data: Dict[str, List]) -> None:
    x = train_data[tag][0]
    scaled_epoch_max = max(train_data['epoch'][1])
    scaled_iteration_max = max(train_data['epoch'][0])
    scaled_x = [it * scaled_epoch_max / scaled_iteration_max for it in x]
    train_data[tag][0] = scaled_x


def scale_time_data(dict_data: Dict[str, List], ref_key: str) -> None:
    elapsed_time = [
        dict_data['wall time'][1][i + 1] - dict_data['wall time'][1][i] for i in
        range(0, len(dict_data['wall time'][1]) - 1)
    ]
    elapsed_time.insert(0, elapsed_time[0])
    dict_data['time (s)'] = [elapsed_time, dict_data[ref_key][0]]
    del dict_data['epoch']
    del dict_data['wall time']

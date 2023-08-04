import argparse
from pathlib import Path

import plot.plot_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path,
                        help='Path to the model directory for name reference (Not a PTH file).', required=True)
    parser.add_argument('--event_runs_dir', type=Path, default=Path('./runs'),
                        help='Path to the runs folder, aka the tensorboard event dir.')
    args = parser.parse_args()
    model_dir: Path = args.model_dir
    event_runs_dir: Path = args.event_runs_dir
    return model_dir, event_runs_dir


def main():
    model_dir, event_runs_dir = parse_args()
    plot.plot_history.plot_training_results(model_dir, event_runs_dir)


if __name__ == "__main__":
    main()

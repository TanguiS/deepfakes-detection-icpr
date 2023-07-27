import argparse
import base64
import csv
import time
from pathlib import Path
from typing import Generator, Tuple, List

import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from evaluation import util
from evaluation.Evaluator import ModelEvaluator
from isplutils import split
from architectures.fornet import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsets', type=str, help='Testing datasets', nargs='+', choices=split.available_datasets,
                        required=True)
    parser.add_argument('--subject_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces program ' +
                             ' for TanguiS deepfacelab project '
                             'Required for training/validating on the SUBJECT dataset.')
    parser.add_argument('--subject_root_dir', type=str, action='store',
                        help='Path to the subject root directory. '
                             'Required for training/validating on the SUBJECT dataset.')
    parser.add_argument('--model_path', type=Path, help='Full path of the trained model', required=True)
    args = parser.parse_args()
    model_path: Path = args.model_path
    test_sets = args.testsets
    subject_df = args.subject_df_path
    subject_root_dir = args.subject_root_dir
    return model_path, subject_df, subject_root_dir, test_sets


def init_evaluation(model_path: Path, net_name: str, subject_df: str, subject_root_dir: str, test_sets: List[str]):
    print('Loading model...')
    net = util.load_model(model_path, net_name)
    print('Model loaded!')
    splits = split.make_splits(
        dfdc_df="", ffpp_df="", dfdc_dir="", ffpp_dir="",
        subject_df=subject_df, subject_dir=subject_root_dir,
        dbs={'train': test_sets, 'val': test_sets, 'test': test_sets}
    )
    test_dfs = [splits['test'][db][0] for db in splits['test']][0]
    return net, test_dfs


def frame_to_base64(frame_path: Path) -> str:
    image = cv2.imread(str(frame_path))
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return base64_image


def test_frames_iterator(df: DataFrame, root: Path) -> Generator[Tuple[str, bool], None, None]:
    for frame_line in df.iterrows():
        yield frame_to_base64(root.joinpath(frame_line[0])), frame_line[1]['label']


def evaluate(
        model_path: str, net: FeatureExtractor, subject_root_dir: str, test_dfs: DataFrame
) -> Tuple[List[int], List[int], float, List[float], List[float]]:
    evaluator = ModelEvaluator(net, Path(model_path))
    count = 0
    sum_time = 0
    y_true = []
    y_pred = []
    yhat_0 = []
    yhat_1 = []

    print("Evaluating...")
    for frame, label in tqdm(test_frames_iterator(test_dfs, Path(subject_root_dir))):
        start = time.time()
        yhat = evaluator.evaluate(frame)
        end = time.time()

        count += 1
        sum_time += (end - start)

        y_true.append(int(label))
        y_pred.append(int(yhat[0] > 0.5))

        yhat_0.append(yhat[0])
        yhat_1.append(yhat[1])

    return y_true, y_pred, sum_time / count, yhat_0, yhat_1


def save_yhat(yhat_0: List[float], yhat_1: List[float], sum_time: float, csv_path: Path) -> None:
    data = list(zip(yhat_0, yhat_1))

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        csv_writer.writerow(['Yhat Fake', 'Yhat Real'])
        csv_writer.writerows(data)
        csv_writer.writerow(['', '', 'Sum Time', sum_time])


def plot_confusion_matrix(y_true: list, y_pred: list, arch: str, shape: Tuple[int, int, int]) -> None:
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {arch} - {shape}')
    plt.show()


def main():
    model_path, subject_df, subject_root_dir, test_sets = parse_args()
    csv_path = model_path.with_name('yhat.csv')

    face_policy, patch_size, net_name, model_name = util.read_models_information(model_path)

    net, test_dfs = init_evaluation(model_path, net_name, subject_df, subject_root_dir, test_sets)

    y_true, y_pred, avg_time, yhat_0, yhat_1 = evaluate(str(model_path), net, subject_root_dir, test_dfs)

    plot_confusion_matrix(y_true, y_pred, net_name, (256, 256, 3))

    save_yhat(yhat_0, yhat_1, avg_time, csv_path)


if __name__ == "__main__":
    main()

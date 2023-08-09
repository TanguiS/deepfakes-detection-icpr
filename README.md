# Video Face Manipulation Detection Through Ensemble of CNNs - Adaptation


This is not the official repository of **Video Face Manipulation Detection Through Ensemble of CNNs** :
[Official Repository](https://github.com/polimi-ispl/icpr2020dfdc).

This repository is an adaptation to the official repo in order to use your own Dataset instead of the ones used for the challenge.

## Getting started

### Prerequisites

- Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Environment creation :

```bash
$ conda create -n icpr python=3.8

$ conda activate icpr

# Either
$ pip install -r requirements.txt

# Or
$ conda install -c conda-forge --file requirements.txt
```

- **DataFrame** requirement : (two possibilities)
  - Use their pre-process scripts on your dataset, in that case you will have to create a file *metadata.json* which 
is a reference to the path and label of the *real* and fake *frames* (no available scripts to do so) :
  - If you used you own face extraction algorithm or want to, here are the pandas DataFrame column requirements :
    - **index** of the dataframe: *relative paths* to the frames from the location of the dataset root directory.
    - *label* column with **True** or **False** values respectively for **Fake** and **Real** frames.
    - **top**, **bottom**, **left** and **right** columns for the box of the extracted faces (If the frames are already extracted at their raw form, it will be the size of their size (0, 0, 512, 512) for instance).

*metadata.json* example of a fake video (split is always "train") and a real video :

```json
{
  "path/to/the/fake/videos/fake_video_1.mp4": {
    "label": "FAKE", "split": "train", "original": "path/to/the/real/videos/original_video_1.mp4"
  },
  "path/to/the/real/videos/original_video_1.mp4": {"label": "REAL", "split": "train"}
}
```

preprocess example using their [scripts](./scripts/make_dataset.sh), use it like if it was the **dfdc** dataset:

```bash
# Indexing videos
# Usage
$ python index_dfdc.py -h

# The metadata.json needs to be in the dataset dir
$ python index_dfdc.py \
--source "/path/to/your/dataset"
--videodataset "path/to/your/dataframe.pkl"
```

```bash
# Extracting faces
# Usage
$ python extract_faces.py -h
```

### Training

Only the [train_binclass.py](train_binclass.py) script was tested, if the [train_triplet.py](train_triplet.py) script you juste need to follow the same steps that I adapted in the binclass script.

The **distribution** that will be used for your own dataset can be changed by modifying the code in the [split.py](isplutils/split.py) file (follow the instruction line 28).

Available model architecture or **net** : for binclass, [fornet.py](architectures/fornet.py) / for triplet, [tripletnet.py](architectures/tripletnet.py).

Example of training command on *Xception* architecture :

```bash
# Usage
$ python train_binclass.py -h

# Training an Xception model
$ python train_binclass.py \
--net "Xception" \
--traindb "subject-85-10-5" \
--valdb "subject-85-10-5" \
--subject_df_path "path/to/your/dataframe.pkl" \
--subject_root_dir "path/to/your/dataset" \
--size 256 \
-- batch 128 \
--maxiter 60000 \
--logint 10 \
--log_dir "path/to/your/output/logs" \
--models_dir "path/to/your/output/models"
```

You can dynamically monitor the training using the *tensorboard* lib :

```bash
$ tensorboard --logdir=path/to/your/output/logs
```

Or using my plot method, [plot_model_history.py](plot_model_history.py) (mainly for presentation purposes) :

```bash
# Usage
$ python plot_model_history.py -h

# Plot training history on Xception architecture
$ python plot_model_history.py \
--model_dir "path/to/your/output/models/**/Xception*" \
--event_runs_dir "path/yo/your/output/logs/**/Xception*"
```

### Testing

Their **testing** script, [test_model.py](test_model.py), will create a pickle file with new column that refers to the achieved score using the *test batch* (it also add a column for binary, categorical column)

Example of testing command on *Xception* architecture :

```bash
$ python test_model.py \
--testsets "subject-85-10-5" \
--testsplits 'test' \
--subject_df_path "path/to/your/dataframe.pkl" \
--subject_root_dir "path/to/your/dataset" \
--model_path "path/to/your/output/models/**/bestval.pth" \
--results_dir "path/to/your/output/test/results/" \
--override
```

My **testing** script, [evaluate_model.py](evaluate_model.py), will show you two matplotlib graph without auto saving. It will create a **yhat.csv** in the same folder as the weights are stored, which save the yhat results and the mean time to evaluate a frame at the end of the file. It will also display a **confusion matrix** which will not be automatically saved.

```bash
# Usage 
$ python evaluate_model.py -h

# Evaluating
$ python evaluate_model.py \
--testsets "subject-85-10-5" \
--subject_df_path "path/to/your/dataframe.pkl" \
--subject_root_dir "path/to/your/dataset" \
--model_path "path/to/your/output/models/**/bestval.pth"
```

## Output Example

My training histories and testing can be found in the [results'](results) folder.

My tensorboard histories can also be found in the [runs'](runs) folder.

## References
Plain text:
```
N. Bonettini, E. D. Cannas, S. Mandelli, L. Bondi, P. Bestagini and S. Tubaro, "Video Face Manipulation Detection Through Ensemble of CNNs," 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 5012-5019, doi: 10.1109/ICPR48806.2021.9412711.
```

Bibtex:

```bibtex
@INPROCEEDINGS{9412711,
  author={Bonettini, Nicolò and Cannas, Edoardo Daniele and Mandelli, Sara and Bondi, Luca and Bestagini, Paolo and Tubaro, Stefano},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
  title={Video Face Manipulation Detection Through Ensemble of CNNs}, 
  year={2021},
  volume={},
  number={},
  pages={5012-5019},
  doi={10.1109/ICPR48806.2021.9412711}}
```

## Credits

- Tangui Steimetz, [ENSICAEN](https://www.ensicaen.fr/).
- Guoqiang Li, [MOBAI](https://www.mobai.bio/) - [NTNU](https://www.ntnu.edu/).

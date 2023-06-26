from pathlib import Path

import pandas as pd


dataset = Path('D:\storage-photos\subjects\dataset.pkl')
df = pd.read_pickle(dataset)

dataset = Path('D:\storage-photos\subjects\dataset_face.pkl')
df = pd.read_pickle(dataset)

a = 0

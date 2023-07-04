from pathlib import Path

import pandas as pd
from pandas import DataFrame

dataset = Path('C:\WORK\subjects\dataframe.pkl')
df: DataFrame = pd.read_pickle(dataset)

size = len(df)
left_top = []
right_bottom = []

for i in range(0, size):
    left_top.append(0)
    right_bottom.append(512)

df['left'] = left_top
df['top'] = left_top
df['right'] = right_bottom
df['bottom'] = right_bottom

test = dataset.with_name("dataframe_box.pkl")

df.to_pickle(str(test))

a = 0

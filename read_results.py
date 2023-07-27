from pathlib import Path

import pandas as pd

file = Path('./results/net-EfficientNetAutoAttB4_traindb-subject-85-10-5_face-scale_size-256_seed-41_bestval/subject-85-10-5_test.pkl')

obj = pd.read_pickle(file)

 print(obj)

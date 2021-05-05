from glob import glob
import sys

import pandas as pd

a_csvs = glob(sys.argv[1])

for a_csv in a_csvs:
    path = a_csv
    df = pd.read_csv(a_csv)
    s = df.iloc[:, -1]

    v = 0
    l = []
    for a in s:
        v += a
        l.append(v)

    new_s = pd.Series([0.0] + l)
    new_s.name = 'speed'
    path = path.replace('acc', 'speed', 2)
    new_s.to_csv(path)
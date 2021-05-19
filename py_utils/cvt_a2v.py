from glob import glob
import os
import sys

import pandas as pd

prefix = sys.argv[1][:-1]
a_csvs = glob(sys.argv[1])

for a_csv in a_csvs:
    if os.path.isdir(a_csv):
        continue

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
    i = path.rfind('\\')
    if i == -1:
        print('rfind failed.')
        exit(-1)

    filename = path[i + 1:].replace('acc', 'speed')
    path = prefix + 'speed\\' + filename
    new_s.to_csv(path)
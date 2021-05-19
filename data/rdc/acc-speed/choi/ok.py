from glob import glob
import os
import sys

import pandas as pd

if sys.argv[1][-1] == '*':
    csvs = glob(sys.argv[1])
else:
    csvs = [sys.argv[1]]

for csv in csvs:
    if '.py' in csv:
        continue
        
    if os.path.isdir(csv):
        continue
        
    df = pd.read_csv(csv)

    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    for c in df:
        df[c] = df[c].map(lambda x: x if x >= 0 else 0.0)

    df.to_csv('filtered/' + csv)
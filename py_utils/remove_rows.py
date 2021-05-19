from glob import glob
import pathlib
from pathlib import Path
import sys

import pandas as pd

csvs = glob(sys.argv[1])
csvs = [Path(x) for x in csvs]
window_size = int(sys.argv[2])
dst_path = pathlib.Path(sys.argv[3])

for csv in csvs:
    df = pd.read_csv(csv, index_col=0)
    df = df.loc[window_size:]
    df.reset_index(drop=True, inplace=True)
    df.to_csv(dst_path / csv.name)
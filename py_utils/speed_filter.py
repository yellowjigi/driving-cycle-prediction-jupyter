from glob import glob
import pathlib
import sys

import pandas as pd

if len(sys.argv) < 3:
    py = pathlib.Path(sys.argv[0]).name
    print('Usage: {0} <CSV> <Destination Directory>'.format(py))
    exit(0)

args = sys.argv[1:-1]
files = []
for arg in args:
    files += glob(arg)

files = [pathlib.Path(f) for f in files]
dst_path = pathlib.PurePath(sys.argv[-1])

for f in files:
    if not f.exists():
        print('\'{0}\' does not exist!'.format(f))
        continue

    if f.is_dir():
        print('\'{0}\' is a directory!'.format(f))
        continue

    if f.suffix != '.csv':
        print('\'{0}\' is not a CSV file!'.format(f))
        continue

    csv = f
    df = pd.read_csv(csv, index_col=0)

    df['prediction'] = df['prediction'].map(lambda x: 0.0 if x < 0.25 else x)
    # for c in df:
    #     df[c] = df[c].map(lambda x: 0.0 if x < 0.25 else x)

    df.to_csv(dst_path / csv.name)
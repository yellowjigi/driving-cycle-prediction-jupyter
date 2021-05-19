from glob import glob
import sys

import pandas as pd

pred_csvs = glob(sys.argv[1])
gt_csvs = glob(sys.argv[2])

csvs = zip(sorted(pred_csvs), sorted(gt_csvs))

for pair in csvs:
    df_pred = pd.read_csv(pair[0])
    if 'ARIMA' in pair[0]:
        df_pred = df_pred[5:]

    df_gt = pd.read_csv(pair[1])
    if 'ARIMA' in pair[1]:
        df_gt = df_gt[5:]

    df_merged = pd.concat([df_pred['Soc'], df_gt['Soc']], axis=1)
    df_merged.columns = ['predict', 'label']

    if 'ARIMA' in pair[0]:
        df_merged.reset_index(drop=True, inplace=True)

    dest_path_name = pair[0].replace('pred', 'merged', 1)

    df_merged.to_csv(dest_path_name)
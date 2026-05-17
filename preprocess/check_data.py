import pandas as pd
import os

DATA_DIR = "data_splits_random"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
    FEATURE_COLS = [l.strip() for l in f if l.strip()]
PHYSCHEM_COLS = [c for c in FEATURE_COLS if not c.startswith("fp_") and not c.startswith("fcfp_")]

print(train_df[PHYSCHEM_COLS].describe())

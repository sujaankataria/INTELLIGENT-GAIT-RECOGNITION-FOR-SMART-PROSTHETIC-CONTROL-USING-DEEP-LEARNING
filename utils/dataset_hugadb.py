import os, glob, math, re
import numpy as np
import pandas as pd

class WindowedIMUDataset:
    def __init__(self, csv_files, channels, label_col, window_len, hop_len, norm="zscore_per_channel"):
        self.channels = channels
        self.label_col = label_col
        self.window_len = int(window_len)
        self.hop_len = int(hop_len)
        self.norm = norm
        self.samples = []
        self.label_to_int = {}
        self.int_to_label = {}

        uniq_labels = set()
        for f in csv_files:
            try:
                s = pd.read_csv(f, usecols=[label_col])[label_col].dropna().unique().tolist()
                uniq_labels.update(s)
            except Exception:
                pass
        uniq_labels = sorted(list(uniq_labels))
        self.label_to_int = {lab:i for i,lab in enumerate(uniq_labels)}
        self.int_to_label = {i:lab for lab,i in self.label_to_int.items()}

        for f in csv_files:
            df = pd.read_csv(f)
            cols_needed = channels + ([label_col] if label_col else [])
            df = df.dropna(subset=cols_needed) if cols_needed else df

            X = df[channels].values.astype(np.float32)
            if label_col:
                y_raw = df[label_col].values
                if y_raw.dtype.kind in ("U","S","O"):
                    y = np.array([self.label_to_int.get(v, -1) for v in y_raw], dtype=np.int64)
                else:
                    y = y_raw.astype(np.int64)
            else:
                y = np.zeros(len(df), dtype=np.int64)

            if self.norm == "zscore_per_channel":
                mu = X.mean(axis=0, keepdims=True)
                std = X.std(axis=0, keepdims=True) + 1e-6
                X = (X - mu) / std

            n = len(X)
            pos = 0
            while pos + self.window_len <= n:
                xw = X[pos:pos+self.window_len]
                yw_slice = y[pos:pos+self.window_len]
                binc = np.bincount(yw_slice, minlength=len(self.label_to_int))
                yw = int(np.argmax(binc))
                self.samples.append((xw, yw))
                pos += self.hop_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xw, yw = self.samples[idx]
        return xw, yw

def collect_files(root, pattern="**/*.csv"):
    return glob.glob(os.path.join(root, pattern), recursive=True)

def split_by_subject(files, subject_col=None, train_ids=None, val_ids=None, test_ids=None):
    train_files, val_files, test_files = [], [], []
    for f in files:
        sub = -1
        if subject_col:
            try:
                df_head = pd.read_csv(f, nrows=20)
                if subject_col in df_head.columns:
                    s = df_head[subject_col].dropna()
                    sub = int(s.iloc[0]) if len(s)>0 else -1
            except Exception:
                sub = -1
        if sub == -1:
            m = re.search(r"_([0-9]{2})_", os.path.basename(f))
            if m:
                sub = int(m.group(1))
        if train_ids and sub in train_ids:
            train_files.append(f)
        elif val_ids and sub in val_ids:
            val_files.append(f)
        elif test_ids and sub in test_ids:
            test_files.append(f)
        else:
            train_files.append(f)
    return train_files, val_files, test_files

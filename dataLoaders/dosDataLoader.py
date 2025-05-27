import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StandardizingDosDataset(Dataset):
    def __init__(self, root_dir, skiprows=4, max_rows=None):
        self.samples = []
        self.root_dir = root_dir

        all_data = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            try:
                class_label = int(class_name.split("_")[-1])
            except ValueError:
                continue

            for fname in os.listdir(class_path):
                if fname.endswith(".dat"):
                    fpath = os.path.join(class_path, fname)
                    df = pd.read_csv(fpath, sep='\s+', skiprows=skiprows, header=None)
                    
                    if max_rows:
                        df = df.iloc[:max_rows]

                    all_data.append(df.values)
                    for row in df.values:
                        self.samples.append((row, class_label))

        all_data = np.vstack(all_data)  # [N, D]
        self.mean = torch.tensor(all_data.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(all_data.std(axis=0), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_np, label = self.samples[idx]
        x = torch.tensor(x_np, dtype=torch.float32)
        x = (x - self.mean) / (self.std + 1e-8)  # стандартизация
        return x, label
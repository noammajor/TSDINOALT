import pandas as pd
import torch
import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np

class DataPuller(Dataset):
    def __init__(self, data_dir, split='train', transform=None, batch_size=32, patch_size=16, step_size=12):
        self.data_dir = data_dir
        self.which = split
        self.transform = transform
        self.data_dir = data_dir
        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.patch_size = patch_size
        self.num_patches = batch_size
        self.val_prec = 0.1
        self.test_prec = 0.1
        self.all_map = {'train': [], 'val': [], 'test': []}
        self.window_size = self.num_patches * self.patch_size
        self.step_size = step_size  # monthly data assumed

        #called last
        self.get_data()

    def get_data(self):
        df = pd.read_csv(self.data_dir, parse_dates = ['date'])
        fcols = df.select_dtypes("float").columns.tolist()
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
        icols = df.select_dtypes("integer").columns
        df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
        df.sort_values(by='date', inplace=True)
        input_vars = [col for col in df.columns if col != 'date']
        val_len = int(len(df) * self.val_prec)
        test_len = int(len(df) * self.test_prec)
        train_len = len(df) - val_len - test_len
        df = torch.tensor(df[input_vars].values).float()
        train_df, val_df, test_df = torch.split(df, [train_len, val_len, test_len])
        self.Train_Val_Test_splits['train'].append(train_df)
        self.Train_Val_Test_splits['val'].append(val_df)
        self.Train_Val_Test_splits['test'].append(test_df)
        for split_name in ['train', 'val', 'test']:
            tensor = self.Train_Val_Test_splits[split_name][0]
            num_samples = (tensor.size(0) - self.window_size) // self.step_size # using stride of 1 patch
            print(f"Number of samples in {split_name} set: {num_samples}")
            for i in range(num_samples):
                self.all_map[split_name].append((0, i))
    def __len__(self):
        return len(self.all_map[self.which])
    
    def __getitem__(self, idx):
        file_idx, start_offset = self.all_map[self.which][idx]
        source_data = self.Train_Val_Test_splits[self.which][file_idx]
        start = start_offset * self.step_size
        end = start + self.window_size
        chunk = source_data[start:end] 
        patches_tensor = chunk.view( self.num_patches, self.patch_size, -1).transpose(1, 2)
        if self.transform:
            patches_tensor = self.transform(patches_tensor)
        return patches_tensor
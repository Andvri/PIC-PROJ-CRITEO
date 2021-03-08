import os
import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader



class TextDataset(Dataset):
    """ Text dataset """
    def __init__(self,csv_file,root_dir="",transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dataframe = pd.read_csv(csv_file)
        self.root_dir= root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataframe.iloc[idx]
        return sample
    


import torch
import numpy as np

class DataSet(torch.utils.data.Dataset):

    def __init__(self,
                 data_path: str,
                 label_path: str,):

        self.data_path = data_path
        self.label_path = label_path
        self.load_data()
        

    def load_data(self):

        self.data = np.load(self.data_path)
        self.label = np.load(self.label_path)
        self.size = len(self.label)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple:

        data = self.data[index]
        label = self.label[index]

        return data, label

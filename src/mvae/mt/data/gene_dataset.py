import numpy as np
import pandas as pd
import scipy.io
import torch

from torch.utils.data import Dataset
from typing import List


class GeneDataset(Dataset):

    def __init__(
        self,
        mtx_path: str,
        batch_paths: List[str],
    ):
        self.mtx_path = mtx_path
        self.batch_paths = batch_paths
        self.expression_matrix = self._load_expression_matrix()
        self.batch_indices = self._load_batch_indices()
        self.n_gene = self.expression_matrix.shape[1]
        self.n_batch = [len(np.unique(self.batch_indices[:, i])) for i in range(self.batch_indices.shape[1])]

    def _load_expression_matrix(self):
        return scipy.io.mmread(self.mtx_path).toarray().astype(np.float32).T
    
    def _load_batch_indices(self):
        batch_indices = []
        for path in self.batch_paths:
            df = pd.read_csv(path, sep="\t", header=None)
            batch_indices.append(df.values)
        return np.concatenate(batch_indices, axis=1)

    def __len__(self):
        return self.expression_matrix.shape[0]
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.expression_matrix[idx]).float()
        batch_idx = torch.from_numpy(self.batch_indices[idx]).long()
        return x, batch_idx
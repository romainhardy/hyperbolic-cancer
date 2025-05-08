import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
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
        return scipy.io.mmread(self.mtx_path).astype(np.float32).T.tocsr()
    
    def _load_batch_indices(self):
        batch_indices = []
        for path in self.batch_paths:
            df = pd.read_csv(path, sep="\t", header=None)
            batches = df.values.astype(int)
            for i in range(batches.shape[1]):
                unique_values = np.unique(batches[:, i])
                value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
                mapped_values = np.array([value_to_idx[val] for val in batches[:, i]])
                batch_indices.append(mapped_values.reshape(-1, 1))
        return np.concatenate(batch_indices, axis=1)

    def __len__(self):
        return self.expression_matrix.shape[0]
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.expression_matrix[idx].toarray().squeeze(0)).float()
        batch_idx = torch.from_numpy(self.batch_indices[idx]).long()
        return x, batch_idx
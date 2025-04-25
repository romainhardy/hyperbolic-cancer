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
        preprocessing_options: dict
    ):
        self.mtx_path = mtx_path
        self.batch_paths = batch_paths
        self.preprocessing_options = preprocessing_options

        self.X_r, self.X_p = self._load_expression_matrix()
        self.batch_indices = self._load_batch_indices()
        self.n_gene_r = self.X_r.shape[1]
        self.n_gene_p = self.X_p.shape[1]
        self.n_batch = [len(np.unique(self.batch_indices[:, i])) for i in range(self.batch_indices.shape[1])]

    def _preprocess(self, adata, options):
        X_r = adata.X.copy()
        if options["normalize"]:
            sc.pp.normalize_total(adata)
            X_r = adata.X.copy() # Normalized counts (reconstruction targets)
        if options["log1p"]:
            sc.pp.log1p(adata)
        if options["top_genes"]:
            sc.pp.highly_variable_genes(adata, n_top_genes=options["top_genes"])
            adata = adata[:, adata.var.highly_variable].copy()
        if options["scale"]:
            sc.pp.scale(adata)
        X_p = adata.X.copy() # Preprocessed counts (inputs to the encoder)
        return X_r, X_p

    def _load_expression_matrix(self):
        X = scipy.io.mmread(self.mtx_path).toarray().astype(np.float32).T
        n_cells, n_genes = X.shape
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(n_genes)]
        adata.obs_names = [f"c{i}" for i in range(n_cells)]
        X_r, X_p = self._preprocess(adata, self.preprocessing_options)
        return X_r, X_p
    
    def _load_batch_indices(self):
        batch_indices = []
        for path in self.batch_paths:
            df = pd.read_csv(path, sep="\t", header=None)
            batch_indices.append(df.values)
        return np.concatenate(batch_indices, axis=1)

    def __len__(self):
        return self.X_r.shape[0]
    
    def __getitem__(self, idx):
        x_r = torch.from_numpy(self.X_r[idx]).float()
        x_p = torch.from_numpy(self.X_p[idx]).float()
        batch_idx = torch.from_numpy(self.batch_indices[idx]).long()
        return x_r, x_p, batch_idx
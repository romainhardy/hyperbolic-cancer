import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


TRAIN_MTX_PATH = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_counts_train.mtx"
VALID_MTX_PATH = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_counts_valid.mtx"
TEST_MTX_PATH  = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_counts_test.mtx"

CELL_TYPE_TRAIN_PATH = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_cell_type_train.tsv"
CELL_TYPE_VALID_PATH = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_cell_type_valid.tsv"
CELL_TYPE_TEST_PATH = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_cell_type_test.tsv"

CLUSTER_ASSIGNMENTS_TRAIN_PATH = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_cluster_assignments_train.npy"
CLUSTER_ASSIGNMENTS_VALID_PATH = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_cluster_assignments_valid.npy"
CLUSTER_ASSIGNMENTS_TEST_PATH  = "/home/romainlhardy/code/hyperbolic-cancer/data/cellxgene/climb_cluster_assignments_test.npy"

class MoreComplexMLP(nn.Module):
    
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_rate=0.5):
        super(MoreComplexMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)  
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out


def prepare_mtx(path):
    X = scipy.io.mmread(path).astype(np.float32).T.tocsr()
    n_cells, n_genes = X.shape
    adata = ad.AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False)
    X = adata.X.copy()
    return X


def sample_random_mixture(num_clusters, target_size):
    random_values = np.random.rand(num_clusters)
    total_sum = np.sum(random_values)
    if total_sum == 0:
        return np.ones(num_clusters) / num_clusters
    mixture_vector = np.clip(random_values / total_sum, 0.0, 1.0)
    mixture_vector = (mixture_vector * target_size).astype(int)
    return mixture_vector


def run_experiment(X_train, y_train, cluster_assignments_train, X_valid, y_valid, X_test, y_test, num_clusters):
    # Sample random mixture of clusters
    target_size = 100000
    # mixture_vector = (np.array([0.12743568, 0.05945665, 0.10082823, 0.01252351, 0.010723  ,
    #    0.05197455, 0.02300644, 0.05981675, 0.13099668, 0.13079662,
    #    0.06441804, 0.02544713, 0.00044012, 0.01348378, 0.04117153,
    #    0.14748129]) * target_size).astype(int)
    mixture_vector = (np.array([0.03333333, 0.03333333, 0.03333333, 0.03333333, 0.03333333,
       0.03333333, 0.03333333, 0.03333333, 0.03333333, 0.03333333,
       0.03333333, 0.03333333, 0.03333333, 0.03333333, 0.03333333,
       0.5       ]) * target_size).astype(int)
    # mixture_vector = sample_random_mixture(num_clusters, target_size)
    X_samples = []
    y_samples = []
    for i in range(num_clusters):
        n = mixture_vector[i]
        X_subset = X_train[cluster_assignments_train == i].toarray()
        y_subset = y_train[cluster_assignments_train == i]
        indices = np.random.choice(np.arange(len(X_subset)), size=n, replace=True)
        X_samples.append(X_subset[indices])
        y_samples.append(y_subset[indices])
    X_train = np.concatenate(X_samples, axis=0)
    y_train = np.concatenate(y_samples, axis=0)
    # indices = np.random.choice(np.arange(len(y_train)), size=target_size, replace=True)
    # X_train = X_train[indices].toarray()
    # y_train = y_train[indices]

    print(len(np.unique(y_train)), len(np.unique(y_valid)))
    
    # Datasets and dataloaders
    dataset_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    dataset_valid = TensorDataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long())
    dataset_test  = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    dataloader_train = DataLoader(dataset_train, batch_size=256, num_workers=4, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=256, num_workers=4, shuffle=False)
    dataloader_test  = DataLoader(dataset_test, batch_size=256, num_workers=4, shuffle=False)

    # Model definition
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_valid))
    hidden_size1 = 512
    hidden_size2 = 256
    dropout_rate = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoreComplexMLP(input_size, hidden_size1, hidden_size2, num_classes, dropout_rate)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10

    # Training loop (with validation)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader_train)

        # Validation
        model.eval()
        valid_loss = 0.0
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader_valid:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted.cpu().numpy())
                true_labels.append(labels.cpu().numpy())
        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        avg_valid_loss = valid_loss / len(dataloader_valid)
        valid_accuracy = 100.0 * (pred_labels == true_labels).mean()
        valid_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

        # Test
        model.eval()
        test_loss = 0.0
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader_test:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted.cpu().numpy())
                true_labels.append(labels.cpu().numpy())
        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        avg_test_loss = test_loss / len(dataloader_test)
        test_accuracy = 100.0 * (pred_labels == true_labels).mean()
        test_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

        print(f"epoch {epoch + 1:03d} / {num_epochs:03d} || loss: {average_loss:.4f} || " + \
              f"valid_loss: {avg_valid_loss:.4f} || valid_accuracy: {valid_accuracy:.2f}% || valid_f1: {valid_f1:.4f} || " + \
              f"test_loss: {avg_test_loss:.4f} || test_accuracy: {test_accuracy:.2f}% || test_f1: {test_f1:.4f}")

    return valid_accuracy, valid_loss, test_accuracy, test_loss


def main():
    # Load data
    X_train = prepare_mtx(TRAIN_MTX_PATH)
    X_valid = prepare_mtx(VALID_MTX_PATH).todense()
    X_test  = prepare_mtx(TEST_MTX_PATH).todense()

    # Load cell type labels
    y_train = pd.read_csv(CELL_TYPE_TRAIN_PATH, sep="\t", header=None).values[:, 0].astype(int)
    y_valid = pd.read_csv(CELL_TYPE_VALID_PATH, sep="\t", header=None).values[:, 0].astype(int)
    y_test  = pd.read_csv(CELL_TYPE_TEST_PATH, sep="\t", header=None).values[:, 0].astype(int)

    # CLIMB cluster assignments
    cluster_assignments_train = np.load(CLUSTER_ASSIGNMENTS_TRAIN_PATH)
    cluster_assignments_valid = np.load(CLUSTER_ASSIGNMENTS_VALID_PATH)

    assert X_train.shape[0] == y_train.shape[0] == cluster_assignments_train.shape[0]
    assert X_valid.shape[0] == y_valid.shape[0] == cluster_assignments_valid.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    num_clusters = len(np.unique(cluster_assignments_train))
    print(f"Number of clusters: {num_clusters}")

    valid_accuracy, valid_loss, test_accuracy, test_loss = run_experiment(X_train, y_train, cluster_assignments_train, X_valid, y_valid, X_test, y_test, num_clusters)

if __name__ == "__main__":
    main()

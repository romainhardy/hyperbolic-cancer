# hyperbolic-cancer

## Abstract
The characterization of diverse cellular states within tumors is pivotal for advancing our understanding of cancer progression. Single-cell RNA sequencing (scRNA-seq) provides unprecedented resolution into these states; however, prevailing dimensionality reduction techniques predominantly operate within Euclidean space, which may not optimally capture the intricate, often non-Euclidean, structures inherent in biological data. To overcome this limitation, we introduce CURVE (\textbf{C}ellular \textbf{U}nderstanding via \textbf{R}iemannian \textbf{V}ariational \textbf{E}ncoding), a novel variational autoencoder architecture. CURVE learns representations on product manifolds, thereby providing a more geometrically expressive framework than conventional Euclidean approaches. Our findings demonstrate that CURVE embeddings effectively encode meaningful biological signals, such as cell type and tumor stage. We establish the efficacy of CURVE both as a powerful visualization tool and through rigorous quantitative comparisons of its embedding quality against widely-used dimensionality reduction methods, including UMAP and $t$-SNE.

## Training CURVE
To train a CURVE model, create a configuration file (see the `config` folder for examples) and run the following command:
```
python3 -m src.train --config <path/to/config/file>
```

## Code
The model and dataset implementations are located at `src/mvae/mt/mvae/models/gene_vae.py` and `src/mvae/mt/data/gene_dataset.py`, respectively.

## Acknowledgements
This repository borrows from the work of [Skopec et al. (2019)](https://github.com/oskopek/mvae).

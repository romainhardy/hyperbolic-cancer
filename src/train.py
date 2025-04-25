import pytorch_lightning as pl
import torch
import yaml

from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.mvae.mt.data import GeneDataset
from src.lightning import GeneModule
from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size, num_workers, mode="train"):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=mode == "train",
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def main():
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/romainlhardy/code/hyperbolic-cancer/configs/mvae/mvae.yaml",
        type=str,
        required=True
    )
    args = parser.parse_args()
    
    # Configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Dataset
    dataset = GeneDataset(**config["data"]["options"])
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        mode="train"
    )

    # Lightning module
    config["lightning"]["model"]["options"]["n_gene_r"] = dataset.n_gene_r
    config["lightning"]["model"]["options"]["n_gene_p"] = dataset.n_gene_p
    config["lightning"]["model"]["options"]["n_batch"] = dataset.n_batch
    module = GeneModule(config)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        save_weights_only=False,
        dirpath=config["output_dir"],
        filename=f"{config.get('experiment', 'mvae')}_{{epoch:02d}}",
        save_top_k=1,
        every_n_epochs=1,
        verbose=1,
    )
    
    # Logger
    logger = WandbLogger(
        project="hyperbolic-cancer",
        name=config.get("experiment", "mvae")
    )
    
    # Trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint],
        logger=logger,
        **config["trainer"],
    )
    
    # Train the model
    trainer.fit(
        module,
        train_dataloaders=dataloader,
        ckpt_path=config.get("checkpoint_path", None),
    )


if __name__ == "__main__":
    main()
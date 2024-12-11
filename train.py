import os
import hydra
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import wandb
import flatdict
from omegaconf import OmegaConf

from src.models.model_util import load_model
from src.datasets.dataset_util import load_dataset

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):

    # Configurazione del seed
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    # Caricamento del modello e dataset
    model = load_model(cfg)
    train, val, test = load_dataset(cfg)

    # DataLoader per train, val e test
    train_loader = DataLoader(
        train, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, persistent_workers=True
    )
    val_loader = DataLoader(
        val, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, persistent_workers=True
    )
    test_loader = DataLoader(
        test, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, persistent_workers=True
    )

    # Configurazione WandbLogger
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project
    )


    hyper = OmegaConf.to_container(cfg, resolve=True)
    hyperparameters =  dict(flatdict.FlatDict(hyper, delimiter="/"))

    wandb.config.update(hyperparameters)  # Salva tutta la configurazione su WandB
    wandb_logger = WandbLogger(log_model=True)  # Abilita il logging del modello

    # Configurazione del trainer
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.max_epochs,
        logger=wandb_logger
    )

    # Training e test
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
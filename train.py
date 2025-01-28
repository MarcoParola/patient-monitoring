import os
import hydra
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
    # Set up seed for reproducibility
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    pl.seed_everything(cfg.train.seed)  # Use PyTorch Lightning's seed_everything

    # Load model and dataset
    model = load_model(cfg)
    train, val, test = load_dataset(cfg)
        
    # DataLoader configuration with device-aware settings
    dataloader_kwargs = {
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.train.num_workers,
        'persistent_workers': True,
        'pin_memory': True if cfg.train.accelerator == "gpu" else False,
    }

    train_loader = DataLoader(train, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **dataloader_kwargs)

    # WandB configuration
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project
    )

    hyper = OmegaConf.to_container(cfg, resolve=True)
    hyperparameters = dict(flatdict.FlatDict(hyper, delimiter="/"))
    wandb.config.update(hyperparameters)
    wandb_logger = WandbLogger(log_model=True)

    # Trainer configuration with device handling
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.max_epochs,
        callbacks=model.configure_callbacks(cfg.train.early_stopping, cfg.train.patience),
        logger=wandb_logger,
        deterministic=True,  # Ensure reproducibility
        precision='16-mixed' if cfg.train.accelerator == "gpu" else '32'
    )

    # Training and testing
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    wandb.finish()

if __name__ == "__main__":
    main()
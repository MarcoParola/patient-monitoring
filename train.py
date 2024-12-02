import os
import hydra
import torch
import pytorch_lightning as pl

from src.models.model_utils import get_model
from src.data.data_utils import get_data


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    print(cfg.pretty())

    model = get_model(cfg)
    train, val, test = get_data(cfg)
    # TODO define dataloader or datamodule

    trainer = pl.Trainer() # TODO

if __name__ == "__main__":
    main()
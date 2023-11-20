import os
import hydra
import torch
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import NearestNeighbors
from src.models.triplet_net import TripletNetModule
from src.datasets.triplet_datamodule import TripletDataModule
from src.loss_log import LossLogCallback

from src.utils import *


@hydra.main(version_base=None, config_path="./config", config_name="config_projection")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.append(get_early_stopping(cfg))
    callbacks.append(LossLogCallback(cfg))
    loggers = get_loggers(cfg)

    torch.set_float32_matmul_precision("high")

    # model
    model = TripletNetModule(
        lr = cfg.train.lr,
        max_epochs = cfg.train.max_epochs
    )

    # datasets and transformations
    data = TripletDataModule(
        train=cfg.dataset.train,
        val=cfg.dataset.val,
        test=cfg.dataset.test,
        features=cfg.features_extractor.features_dataset,
        ranking=cfg.features_extractor.ranking,
        img_dataset=cfg.features_extractor.original,
        batch_size=cfg.train.batch_size
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, 
        monitor="val_loss",
        mode="min"
        )
    callbacks.append(checkpoint_callback)
    # training
    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
        #gradient_clip_val=0.1, 
        #gradient_clip_algorithm="value"
    )
    trainer.fit(model, data)

    trainer.test(dataloaders=data.test_dataloader(), ckpt_path='best')


if __name__ == "__main__":
    main()
import os
import hydra
import torch
import pytorch_lightning
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.image_autoencoder import ImageAutoencoder
from src.datasets.classifier_datamodule import OralClassificationDataModule
from src.loss_log import LossLogCallback
from src.utils import *


@hydra.main(version_base=None, config_path="./config", config_name="config_autoencoder")
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
    model = ImageAutoencoder(
        latent_dim=cfg.train.latent_dim, 
        lr = cfg.train.lr,
        max_epochs = cfg.train.max_epochs
    )

    # datasets and transformations
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    data = OralClassificationDataModule(
        train=cfg.dataset.train,
        val=cfg.dataset.val,
        test=cfg.dataset.test,
        batch_size=cfg.train.batch_size,
        train_transform = train_img_tranform,
        val_transform = val_img_tranform,
        test_transform = test_img_tranform,
        transform = img_tranform,
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
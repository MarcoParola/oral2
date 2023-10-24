import os
import hydra
import torch
import pytorch_lightning
from sklearn.metrics import classification_report
import numpy as np

from src.models.classifier import OralClassifierModule
from src.datasets.datamodule import OralClassificationDataModule
from src.loss_log import LossLogCallback


from test import predict
from src.utils import *

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.append(get_early_stopping(cfg))
    callbacks.append(LossLogCallback())
    loggers = get_loggers(cfg)

    # model
    model = OralClassifierModule(
        model=cfg.model.name,
        weights=cfg.model.weights,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.lr,
        #max_epochs = cfg.train.max_epochs
    )

    # datasets and transformations
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    data = OralClassificationDataModule(
        train=cfg.dataset.train,
        val=cfg.dataset.val,
        test=cfg.dataset.test,
        crop = cfg.dataset.crop,
        batch_size=cfg.train.batch_size,
        train_transform = train_img_tranform,
        val_transform = val_img_tranform,
        test_transform = test_img_tranform,
        transform = img_tranform,
    )

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

    #prediction
    predict(trainer, model, data, cfg.saliency.method)



if __name__ == "__main__":
    main()
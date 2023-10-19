import os
import hydra
import torch
import pytorch_lightning
from sklearn.metrics import classification_report
import numpy as np

from src.models.classifier import OralClassifierModule
from src.datasets.datamodule import OralClassificationDataModule
from src.loss_log import LossLogCallback

from src.utils import *

def predict(trainer, model, data):
    predictions = trainer.predict(model, data)  # TODO: inferenza su piu devices
    predictions = torch.cat(predictions, dim=0)
    predictions = torch.argmax(predictions, dim=1)
    gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)

    print(classification_report(gt, predictions))

    class_names = np.array(['Neoplastic', 'Aphthous', 'Traumatic'])
    log_dir = 'logs/oral/' + get_last_version('logs/oral')
    log_confusion_matrix(gt, predictions, classes=class_names, log_dir=log_dir)  # TODO cambia nome, perchè loggo anche acc

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    # to test is needed: trainer, model and data

    # trainer
    trainer = pytorch_lightning.Trainer(
            logger=get_loggers(cfg),
           # callbacks=callbacks,  shouldn't need callbacks in test
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            log_every_n_steps=1,
            max_epochs=cfg.train.max_epochs,
            #gradient_clip_val=0.1,
            #gradient_clip_algorithm="value"
        )

    # model
    # get the model already trained from checkpoints
    # TODO sistemare il path, ora sto prendenddo sempre lo stesso checkpoint
    model = OralClassifierModule.load_from_checkpoint("logs/oral/version_0/checkpoints/epoch=0-step=7.ckpt")
    model.eval()

    # data
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    data = OralClassificationDataModule(
        train=cfg.dataset.train,
        val=cfg.dataset.val,
        test=cfg.dataset.test,
        batch_size=cfg.train.batch_size,
        train_transform=train_img_tranform,
        val_transform=val_img_tranform,
        test_transform=test_img_tranform,
        transform=img_tranform,
    )

    predict(trainer, model, data)

if __name__ == "__main__":
    main()
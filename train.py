import os
import hydra
import torch
import pytorch_lightning
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import re
import tfplot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from src.classifier import OralClassifierModule
from src.datamodule import OralClassificationDataModule

from src.utils import get_loggers, get_early_stopping, get_transformations



import seaborn as sn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter




def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()




@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    
    callbacks = list()
    callbacks.append(get_early_stopping(cfg))
    loggers = get_loggers(cfg)

    # model
    model = OralClassifierModule(
        model=cfg.model.name,
        weights=cfg.model.weights,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.lr,
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

    # training
    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
    )
    trainer.fit(model, data)

    # prediction
    predictions = trainer.predict(model, data)   # TODO: inferenza su piu devices
    predictions = torch.cat(predictions, dim=0)
    predictions = torch.argmax(predictions, dim=1)
    gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)
    
    print(classification_report(gt, predictions))


    writer = SummaryWriter('logs')
    classes = np.array(['Class 0', 'Class 1', 'Class 2'])
    cf_matrix = confusion_matrix(gt, predictions)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    img = sn.heatmap(df_cm, annot=True).get_figure()
    print(type(img))
    writer.add_figure("Confusion matrix", img, 0)

    
    # Define your class labels here
    class_names = np.array(['Class 0', 'Class 1', 'Class 2'])

    # Plot normalized confusion matrix
    plot_confusion_matrix(gt, predictions, classes=class_names, normalize=True)
    plt.show()




    

if __name__ == "__main__":
    main()

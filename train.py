import os
import hydra
import torch
import pytorch_lightning
from sklearn.metrics import classification_report

from src.classifier import OralClassifierModule
from src.datamodule import OralClassificationDataModule

from src.utils import get_loggers, get_early_stopping, get_transformations

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
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations()
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

    # log classification report
    loggers[0].experiment.log({"classification_report": classification_report(gt, predictions, output_dict=True)})

    if cfg.train.save_path != "":
        trainer.save_checkpoint(cfg.train.save_path)
    

if __name__ == "__main__":
    main()



import os
import hydra
import torch
import torchvision
import pytorch_lightning
from sklearn.metrics import classification_report

from src.classifier import OralClassifierModule
from src.datamodule import OralClassificationDataModule

from src.utils import get_loggers, get_early_stopping

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

    
    model = OralClassifierModule(
        model=cfg.model.name,
        weights=cfg.model.weights,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.lr,
    )
    data = OralClassificationDataModule(
        train=cfg.dataset.train,
        val=cfg.dataset.val,
        test=cfg.dataset.test,
        batch_size=cfg.train.batch_size,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
            torchvision.transforms.CenterCrop(cfg.dataset.resize),
            torchvision.transforms.ToTensor(),
        ]),
    )

    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
        check_val_every_n_epoch=2
    )
    
    trainer.fit(model, data)
    predictions = trainer.predict(model, data)   # TODO: inferenza su piu devices
    
    predictions = torch.cat(predictions, dim=0)
    predictions = torch.argmax(predictions, dim=1)

    gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)
    
    print(classification_report(gt, predictions))

    if cfg.train.save_path != "":
        trainer.save_checkpoint(cfg.train.save_path)
    

if __name__ == "__main__":
    main()

import os
import hydra
import torch
import torchvision
import pytorch_lightning

from src.classifier import OralClassifierModule
from src.datamodule import OralClassificationDataModule

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    
    torch.manual_seed(cfg.train.seed)

    loggers = list()
    callbacks = list()
    if cfg.log.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)
    
    if cfg.log.tensorboard:
        from pytorch_lightning.loggers import TensorBoardLogger
        tensorboard_logger = TensorBoardLogger("logs", name="oral")
        loggers.append(tensorboard_logger)

    model = OralClassifierModule(
        model=cfg.model.name,
        weights=cfg.model.weights,
        num_classes=cfg.model.num_classes,
        lr=cfg.train.lr,
    )
    data = OralClassificationDataModule(
        train=cfg.dataset.train,
        test=cfg.dataset.test,
        batch_size=cfg.train.batch_size,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
            torchvision.transforms.CenterCrop(cfg.dataset.resize),
            torchvision.transforms.ToTensor(),
        ])
    )

    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
    )
    
    trainer.fit(model, data)

    if cfg.train.save_path != "":
        trainer.save_checkpoint(cfg.train.save_path)


if __name__ == "__main__":
    main()

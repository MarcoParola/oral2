import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from src.utils import *

class LossLogCallback(pl.Callback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_fit_start(self, trainer, pl_module):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())

    def on_train_end(self, trainer, pl_module):
        path = self.cfg.log.path + self.cfg.log.dir 
        log_dir = path + '/' + get_last_version(path)
        writer = SummaryWriter(log_dir=log_dir)
        for i in range(0, len(self.train_losses)):
            writer.add_scalars('train_val_loss', {'train':self.train_losses[i],
                                    'val':self.val_losses[i]}, i)
        writer.close()

        

import torch
import torchvision
from pytorch_lightning import LightningModule
import tensorboard as tb

class TripletNetModule(LightningModule):

    def __init__(self, lr = 0.0004, max_epochs = 100, frozen_layers=-1):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Linear(64, 64)
        
        self.loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val") 
        
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _common_step(self, batch, batch_idx, stage):
        anchor, positive, negative = batch
        new_anchor = self(anchor)
        new_positive = self(positive)
        new_negative = self(negative)
        new_loss = self.loss(new_anchor,new_positive,new_negative)
        self.log(f"{stage}_loss", new_loss, on_step=True, on_epoch=True)
        return new_loss
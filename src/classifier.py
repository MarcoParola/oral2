import torch
import torchvision
from pytorch_lightning import LightningModule

class OralClassifierModule(LightningModule):

    def __init__(self, model, weights, num_classes, lr=10e-3):
        super().__init__()
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        
        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)

        self.model = getattr(torchvision.models, model)(weights=weights)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.preprocess = weights.transforms()

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, label = batch
        x = self.preprocess(img)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _common_step(self, batch, batch_idx, stage):
        img, label = batch
        x = self.preprocess(img)
        y_hat = self(x)
        loss = self.loss(y_hat, label)
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss

import torch
import torchvision
from pytorch_lightning import LightningModule
import tensorboard as tb
from sklearn.metrics import accuracy_score

class OralClassifierModule(LightningModule):

    def __init__(self, model, weights, num_classes, lr=10e-3, max_epochs = 100, features_size=64, frozen_layers=-1):
        super().__init__()
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        
        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)

        self.model = getattr(torchvision.models, model)(weights=weights)
        
        # Freeze layers
        if frozen_layers == -1:
            layers_number = len(list(self.model.parameters()))
            frozen_layers=int(layers_number * 0.9)

        for idx, param in enumerate(self.model.parameters()):
            if idx < frozen_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self._set_model_classifier(weights_cls, num_classes, features_size)

        self.classifier = self._get_last_part(weights_cls, num_classes, features_size)

        self.model = torch.nn.Sequential(
            self.model, 
            self.classifier
        )

        self.preprocess = weights.transforms(antialias=True)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val") 
        
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
        img, label = batch
        x = self.preprocess(img)
        output = self(x)
        output = output.cpu() 
        label = label.cpu() 
        output = torch.argmax(output, dim=1)
        accuracy = accuracy_score(output, label)
        self.log('test_accuracy', accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, label = batch
        x = self.preprocess(img)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=(self.hparams.lr*0.1))

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss"
        }

        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        img, label = batch
        x = self.preprocess(img)
        y_hat = self(x)
        loss = self.loss(y_hat, label)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def remove_network_end(self):
        self.model = self.model[0]

    def _set_model_classifier(self, weights_cls, num_classes, features_size): 
        weights_cls = str(weights_cls)
        if "ConvNeXt" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Flatten(1),
                torch.nn.LayerNorm(self.model.classifier[2].in_features),

                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[2].in_features, features_size)
            )
        elif "EfficientNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[1].in_features, features_size)
            )
        elif "MobileNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[0].in_features, features_size),
            )
        elif "VGG" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[0].in_features, features_size),
                torch.nn.ReLU(True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(features_size, features_size)
            )
        elif "DenseNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier.in_features, features_size)
            )
        elif "MaxVit" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(self.model.classifier[5].in_features),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[5].in_features, features_size)
            )
        elif "ResNet" in weights_cls or "RegNet" in weights_cls or "GoogLeNet" in weights_cls:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.fc.in_features, features_size)
            )
        elif "Swin" in weights_cls:
            self.model.head = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.head.in_features, features_size)
            )
        elif "ViT" in weights_cls:
            self.model.heads = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.hidden_dim, features_size)
            )

    def _get_last_part(self, weights_cls, num_classes, features_size): 
        weights_cls = str(weights_cls)
        
        if ("ConvNeXt" in weights_cls or "DenseNet" in weights_cls
            or "ResNet" in weights_cls or "RegNet" in weights_cls 
            or "GoogLeNet" in weights_cls or "Swin" in weights_cls
            or "ViT" in weights_cls or "EfficientNet" in weights_cls):
            return torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(features_size, num_classes)
            )
        elif "MobileNet" in weights_cls:
            return torch.nn.Sequential(
                torch.nn.Hardswish(inplace=True),
                torch.nn.Dropout(p=0.5, inplace=True),
                torch.nn.Linear(features_size, num_classes)
            )
        elif "VGG" in weights_cls:
            return torch.nn.Sequential(
                torch.nn.ReLU(True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(features_size, num_classes)
            )
        elif "MaxVit" in weights_cls:
            return torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(features_size, num_classes)
            )
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.dataset import OralClassificationDataset

class OralClassificationDataModule(LightningDataModule):
    def __init__(self, train, test, batch_size=32, train_transform=None, test_transform=None, transform=None):
        super().__init__()
        if train_transform is None:
            train_transform = transform
        if test_transform is None:
            test_transform = transform

        self.train_dataset = OralClassificationDataset(train, transform=train_transform)
        self.test_dataset = OralClassificationDataset(test, transform=test_transform)
        self.batch_size = batch_size
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
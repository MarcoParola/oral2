from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.dataset import OralClassificationDataset

class OralClassificationDataModule(LightningDataModule):
    def __init__(self, train, test, batch_size=32, transform=None):
        super().__init__()
        self.train_dataset = OralClassificationDataset(train, transform=transform)
        self.test_dataset = OralClassificationDataset(test, transform=None)
        self.batch_size = batch_size
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
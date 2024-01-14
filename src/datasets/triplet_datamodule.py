from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.datasets.triplet_dataset import TripletDataset

class TripletDataModule(LightningDataModule):
    def __init__(self, train, val, test, features,ranking, img_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = TripletDataset(train, features, ranking, img_dataset)
        self.val_dataset = TripletDataset(val, features, ranking, img_dataset)
        self.test_dataset = TripletDataset(test, features, ranking, img_dataset)
        self.batch_size = batch_size
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def get_test_dataset(self):
        return self.test_dataset
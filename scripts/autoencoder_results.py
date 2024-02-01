import hydra
import torch
import json
import pandas as pd
import sys
sys.path.append('src')
from models.image_autoencoder import ImageAutoencoder
from models.image_vae import ImageVAE
from utils import *
sys.path.append('src/datasets')
from classifier_dataset import OralClassificationDataset

@hydra.main(version_base=None, config_path="./../config", config_name="config_autoencoder")
def main(cfg):
    if cfg.results.vae:
        model = ImageVAE.load_from_checkpoint(cfg.log.path+cfg.results.checkpoint_path)
    else:
        model = ImageAutoencoder.load_from_checkpoint(cfg.log.path+cfg.results.checkpoint_path)
    
    model.eval()

    img_tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, antialias=True),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])

    dataset = OralClassificationDataset(cfg.dataset.test, img_tranform)

    model = model.to('cuda')

    for i in range(0, dataset.__len__()):
        img,lbl = dataset.__getitem__(i)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(torch.float32)
        img = img.to('cuda')

        feature = model(img)
        if cfg.results.vae:
            feature = feature[0]
        feature = feature.cpu()
        feature = feature.detach()

        img = img.cpu()
        img = img.squeeze(0)
        feature = feature.squeeze(0)
        images_comparison(img, feature, i)

if __name__ == '__main__':
    main()


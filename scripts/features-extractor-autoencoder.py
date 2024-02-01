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
from triplet_dataset import TripletDataset


@hydra.main(version_base=None, config_path="./../config", config_name="config_autoencoder")
def main(cfg):
    if cfg.features_extractor.vae:
        model = ImageVAE.load_from_checkpoint(cfg.log.path+cfg.features_extractor.checkpoint_path)
    else:
        model = ImageAutoencoder.load_from_checkpoint(cfg.log.path+cfg.features_extractor.checkpoint_path)
    model.eval()

    _, _, _, img_tranform = get_transformations(cfg)

    to_get_features_path = cfg.features_extractor.to_get_feature

    dataset = OralClassificationDataset(to_get_features_path, img_tranform)

    model = model.to('cuda')

    features_dataset=[]

    for i in range(0, dataset.__len__()):
        img, lbl = dataset.__getitem__(i)

        img = torch.unsqueeze(img, dim=0)
        img = img.to(torch.float32)
        img = img.to('cuda')

        if cfg.features_extractor.vae:
            encoded1 = model.encoder_block1(img)
            encodedM1, index1 = model.maxpooling_3_2(encoded1)
            encoded2 = model.encoder_block2(encodedM1)
            encodedM2, index2 = model.maxpooling_3_2(encoded2)
            encoded3 = model.encoder_block3(encodedM2)
            encodedM3, index3 = model.maxpooling_2_2(encoded3)
            x = model.encoder_block4(encodedM3)

            mu =  model.layer1(x)
            sigma = torch.exp(model.layer2(x))
            features = mu + sigma*model.N.sample(mu.shape)

        else:
            encoded1 = model.encoder_block1(img)
            encodedM1, index1 = model.maxpooling_3_2(encoded1)
            encoded2 = model.encoder_block2(encodedM1)
            encodedM2, index2 = model.maxpooling_3_2(encoded2)
            encoded3 = model.encoder_block3(encodedM2)
            encodedM3, index3 = model.maxpooling_2_2(encoded3)
            features = model.encoder_block4(encodedM3)

        features = features.cpu()
        features = features.detach()

        features_dataset.append([dataset.get_image_id(i), features.tolist(), lbl])

    features_dataset = pd.DataFrame(features_dataset, columns=['image_id', 'feature', 'type'])
    features_dataset.to_csv(cfg.features_extractor.features_dataset, sep=';', index=False, header=True)

if __name__ == '__main__':
    main()
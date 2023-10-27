import hydra
import torch
import json
import pandas as pd
import sys
sys.path.append('src')
from models.classifier import OralClassifierModule
from utils import *
sys.path.append('src/datasets')
from dataset import OralClassificationDataset


@hydra.main(version_base=None, config_path="./../config", config_name="config")
def main(cfg):
    model = OralClassifierModule.load_from_checkpoint(cfg.log.path+cfg.features_extractor.checkpoint_path)
    model.eval()
    model.remove_network_end()

    _, _, _, img_tranform = get_transformations(cfg)

    oral_ranked_dataset = pd.read_csv(open(cfg.features_extractor.preprocessed_ranking, "r"),
                 sep=';',
                 engine='python')
    
    to_get=[]
    for i in range(0, len(oral_ranked_dataset)):
        if oral_ranked_dataset.loc[i, "case_id"] not in to_get:
            to_get.append(oral_ranked_dataset.loc[i, "case_id"])
        if oral_ranked_dataset.loc[i, "case_id_pos"] not in to_get:
            to_get.append(oral_ranked_dataset.loc[i, "case_id_pos"])
        if oral_ranked_dataset.loc[i, "case_id_neg"] not in to_get:
            to_get.append(oral_ranked_dataset.loc[i, "case_id_neg"])

    with open(cfg.dataset.original, "r") as f:
        original_dataset = json.load(f)

    original_dataset["images"] = [item for item in original_dataset["images"] if str(item["id"]) in to_get]
    assert len(original_dataset["images"]) <= len(to_get) # in to_get could be present the "NotFound" value

    original_dataset["annotations"] = [item for item in original_dataset["annotations"] if str(item["image_id"]) in to_get]

    to_get_features_path = cfg.features_extractor.to_get_feature
    #json.dump(original_dataset, open(os.path.join("./data", "to_get_features_dataset.json"), "w"), indent=2)
    json.dump(original_dataset, open(os.path.join(to_get_features_path.split("/")[1], to_get_features_path.split("/")[2]), "w"), indent=2)

    dataset = OralClassificationDataset(to_get_features_path, img_tranform)

    model = model.to('cuda')

    features_dataset=[]

    for i in range(0, dataset.__len__()):
        img,lbl = dataset.__getitem__(i)

        img = torch.unsqueeze(img, dim=0)
        img = img.to(torch.float32)
        img = img.to('cuda')

        feature = model(img)
        feature = feature.cpu()
        feature = feature.detach()
        features_dataset.append([dataset.get_image_id(i), feature.tolist(), lbl])

    features_dataset = pd.DataFrame(features_dataset, columns=['image_id', 'feature', 'type'])
    features_dataset.to_csv(cfg.features_extractor.features_dataset, sep=';', index=False, header=True)

if __name__ == '__main__':
    main()
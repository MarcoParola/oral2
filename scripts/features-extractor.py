import hydra
import torch
import json
import pandas as pd
import sys
sys.path.append('src')
from models.classifier import OralClassifierModule
from models.triplet_net import TripletNetModule
from utils import *
sys.path.append('src/datasets')
from classifier_dataset import OralClassificationDataset
from triplet_dataset import TripletDataset


@hydra.main(version_base=None, config_path="./../config", config_name="config_projection")
def main(cfg):
    if cfg.features_extractor.classifier:
        feature_before_projection(cfg)
    else:
        feature_after_projection(cfg)



def feature_before_projection(cfg):
    model = OralClassifierModule.load_from_checkpoint(cfg.log.path+cfg.features_extractor.checkpoint_path)
    model.eval()
    model.remove_network_end()

    _, _, _, img_tranform = get_transformations(cfg)

    oral_ranked_dataset = pd.read_csv(open(cfg.features_extractor.preprocessed_ranking, "r"),
                 sep=';',
                 engine='python')
    
    # Retrieve images used in the triplet dataset
    to_get=[]
    for i in range(0, len(oral_ranked_dataset)):
        if oral_ranked_dataset.loc[i, "case_id"] not in to_get:
            to_get.append(oral_ranked_dataset.loc[i, "case_id"])
        if oral_ranked_dataset.loc[i, "case_id_pos"] not in to_get:
            to_get.append(oral_ranked_dataset.loc[i, "case_id_pos"])
        if oral_ranked_dataset.loc[i, "case_id_neg"] not in to_get:
            to_get.append(oral_ranked_dataset.loc[i, "case_id_neg"])

    with open(cfg.features_extractor.original, "r") as f:
        original_dataset = json.load(f)

    # Maintain only images contained in the original dataset 
    original_dataset["images"] = [item for item in original_dataset["images"] if item["id"] in to_get]
    assert len(original_dataset["images"]) <= len(to_get) # in to_get could be present the "NotFound" value

    original_dataset["annotations"] = [item for item in original_dataset["annotations"] if item["image_id"] in to_get]

    # Save the dataset in order to maintain only the subset of the dataset needed
    to_get_features_path = cfg.features_extractor.to_get_feature
    #json.dump(original_dataset, open(os.path.join("./data", "to_get_features_dataset.json"), "w"), indent=2)
    json.dump(original_dataset, open(os.path.join(to_get_features_path.split("/")[1], to_get_features_path.split("/")[2]), "w"), indent=2)

    # Get the features from the model
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

    # Save the feature dataset
    features_dataset = pd.DataFrame(features_dataset, columns=['image_id', 'feature', 'type'])
    features_dataset.to_csv(cfg.features_extractor.features_dataset, sep=';', index=False, header=True)

# Useful for PCA
def feature_after_projection(cfg):
    model = TripletNetModule.load_from_checkpoint(cfg.log.path+cfg.triplet.checkpoint_path)
    model.eval()

    features_dataset = pd.read_csv(
        open(cfg.features_extractor.features_dataset, "r"), 
        sep=';', 
        engine='python'
    )

    ids = list(features_dataset["image_id"])

    features = list(features_dataset["feature"])
    features=[np.array(eval(feature)) for feature in features]
    features = np.array(features)
    features = features.squeeze()
    features = torch.from_numpy(features)
    features = [feature.to(torch.float32) for feature in features]
    features = [feature.requires_grad_() for feature in features]

    lbls = list(features_dataset["type"])

    model = model.to('cuda')

    features_dataset=[]

    for i in range(0, len(ids)):
        feature = features[i]

        feature = feature.to('cuda')

        feature = model(feature)
        feature = feature.cpu()
        feature = feature.detach()

        features_dataset.append([ids[i], feature.tolist(), lbls[i]])

    features_dataset = pd.DataFrame(features_dataset, columns=['image_id', 'feature', 'type'])
    features_dataset.to_csv(cfg.triplet.features_dataset, sep=';', index=False, header=True)

if __name__ == '__main__':
    main()
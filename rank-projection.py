import hydra
import torch
from src.models.triplet_net import TripletNetModule
from src.datasets.triplet_dataset import TripletDataset
from src.utils import *
import shutil

@hydra.main(version_base=None, config_path="./config", config_name="config_projection")
def main(cfg):
    dataset = pd.read_csv(open(cfg.features_extractor.ranking, "r"),
                 sep=';',
                 engine='python')
    references = list(dataset.columns)
    references = references[2:]

    dataset = json.load(open(cfg.features_extractor.original, "r"))
    image_names={}
    for image in dataset["images"]:
        if image["file_name"] in references:
            image_names[image["file_name"]] = image["id"]

    # Missing anchor found..
    cleaned_references=[]
    for item in references:
        if item in list(image_names.keys()):
            cleaned_references.append(item)
    references = cleaned_references

    reference_ids = [image_names[reference] for reference in references]

    test_dataset = TripletDataset(
        cfg.dataset.test, 
        cfg.features_extractor.features_dataset,
        cfg.features_extractor.ranking,
        cfg.features_extractor.original
    )

    gt = test_dataset.get_ids_ranking()

    features_dataset = pd.read_csv(open(cfg.features_extractor.features_dataset, "r"), sep=';', engine='python')
    i = 0
    while i < len(features_dataset):
        if features_dataset.loc[i, "image_id"] not in reference_ids and features_dataset.loc[i, "image_id"] not in gt.keys() :
            features_dataset = features_dataset.drop(i)
        i+=1
    features_dataset.reset_index(inplace = True, drop = True)

    feature_ids = list(features_dataset["image_id"])

    features = list(features_dataset["feature"])
    features=[np.array(eval(feature)) for feature in features]
    features = np.array(features)
    features = features.squeeze()
    features = torch.from_numpy(features)
    features = [feature.to(torch.float32) for feature in features]
    features = [feature.requires_grad_() for feature in features]

    if cfg.triplet.projection:
        model = TripletNetModule.load_from_checkpoint(cfg.log.path+cfg.triplet.checkpoint_path)
        model.eval()
        model = model.to('cuda')

        predictions = []
        for feature in features:
            feature = feature.to('cuda')
            #feature = feature.cpu()
            feature = feature.detach()
            predictions.append(model.forward(feature))

        predictions = [prediction.cpu() for prediction in predictions]
    else: 
        predictions = features


    reference_predictions=[]
    for item in reference_ids:
        index = feature_ids.index(item)
        reference_predictions.append(predictions[index])

    # Missing anchor found..
    k = len(reference_ids)
    #k = get_k()

    knn = {}
    for key in gt.keys():
        index = feature_ids.index(key)
        anchor = predictions[index]
        anchor_id = feature_ids[index]
        others = list(reference_predictions)
        other_ids = list(reference_ids)
        knn[anchor_id] = get_knn(other_ids, others, anchor, k)

    path = cfg.log.path + cfg.log.dir 
    log_dir = path + '/' + get_last_version(path)

    log_compound_metrics(gt, knn, cfg.triplet.projection, log_dir=log_dir)

if __name__ == '__main__':
    main()
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
        cfg.triplet.features_dataset,
        cfg.features_extractor.ranking,
        cfg.features_extractor.original
    )

    gt = test_dataset.get_ids_ranking()

    feature_ids, features, lbls = test_dataset.get_features_dataset()

    # Project all the features
    if cfg.triplet.projection:
        model = TripletNetModule.load_from_checkpoint(cfg.log.path+cfg.triplet.checkpoint_path)
        model.eval()
        model = model.to('cuda')

        predictions = []
        for feature in features:
            feature = feature.to('cuda')
            feature = feature.detach()
            predictions.append(model.forward(feature))

        predictions = [prediction.cpu() for prediction in predictions]
    else: 
        predictions = [feature.detach().cpu() for feature in features]

    # Isolate the reference features
    reference_predictions=[]
    reference_lbls=[]
    for item in reference_ids:
        index = feature_ids.index(item)
        reference_predictions.append(np.array(predictions[index].detach().numpy()))
        reference_lbls.append(lbls[index])

    test_predictions=[]
    test_lbls=[]
    test_ids=list(gt.keys())
    for item in test_ids:
        index = feature_ids.index(item)
        test_predictions.append(np.array(predictions[index].detach().numpy()))
        test_lbls.append(lbls[index])

    # Missing anchor found..
    k = len(reference_ids)
    accuracy = get_knn_classification(reference_predictions, reference_lbls, test_predictions, test_lbls, 5)
    app = get_knn(reference_ids, reference_predictions, test_predictions, k)

    knn = {}
    for i in range(0, len(test_ids)):
        knn[test_ids[i]] = app[i]

    log_dir = cfg.log.path + cfg.triplet.log_dir

    log_compound_metrics(gt, knn, accuracy, cfg.triplet.projection, log_dir=log_dir)

if __name__ == '__main__':
    main()

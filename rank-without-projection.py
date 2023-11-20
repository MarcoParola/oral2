import hydra
from src.datasets.triplet_dataset import TripletDataset
from src.utils import *
import os
import shutil

@hydra.main(version_base=None, config_path="./config", config_name="config_no_projection")
def main(cfg):
    test_dataset = TripletDataset(
        cfg.dataset.test,
        cfg.features_extractor.features_dataset,
        cfg.features_extractor.ranking,
        cfg.features_extractor.original
    )
    gt = test_dataset.get_ids_ranking()
    ids, features = test_dataset.get_features_dataset()

    k = get_k()

    knn = {}
    for key in gt.keys():
        index = ids.index(key)
        anchor = features[index]
        anchor_id = ids[index]
        others = list(features)
        others.pop(index)
        other_ids = list(ids)
        other_ids.pop(index)
        knn[anchor_id] = get_knn(other_ids, others, anchor, k)


    path = cfg.log.path + cfg.log.dir 
    log_dir = path + '/version_' + cfg.log.version
    if os.path.exists(log_dir):
        # If it exists, delete the directory and its contents
        shutil.rmtree(log_dir)

    # Create the directory
    os.makedirs(log_dir)
    log_compound_metrics(gt, knn, log_dir=log_dir)

if __name__ == "__main__":
    main()
import argparse
import json
import os
import pandas as pd

def check_missing_images(original_dataset, ranking):

    anchors = list(ranking.columns)
    # remove 'id_casi' and 'TIPO DI ULCERA'
    anchors.pop(0)
    anchors.pop(0)

    ranked_images=[case_id for case_id in list(ranking['id_casi']) if case_id not in anchors]
    # remove 'id_casi' and '-1'
    ranked_images.pop(0)
    ranked_images.pop(0)

    total_size1 = len(ranked_images)
    total_size2 = len(anchors)

    i=0
    original_images=[]
    for i in range(0, len(original_dataset["images"])):
        if original_dataset["images"][i]["file_name"] in ranked_images:
            ranked_images.remove(original_dataset["images"][i]["file_name"])
        if original_dataset["images"][i]["file_name"] in anchors:
            anchors.remove(original_dataset["images"][i]["file_name"])

    print("Missing images found: " + str(len(ranked_images)) + "/" + str(total_size1))
    print(ranked_images)

    print("Missing anchor images found: " + str(len(anchors)) + "/" + str(total_size2))
    print(anchors)


if __name__ == '__main__':
    '''script to check for missing image files
    expects a cli parameters:
        "--ranked_dataset" valued as csv for the ranked dataset
         "--original_dataset" valued by the json file path of the original dataset
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset", type=str, required=True)
    parser.add_argument("--ranking", type=str, required=True)
    args = parser.parse_args()
    
    #./data/dataset.json 
    with open(args.original_dataset, "r") as f:
        original_dataset = json.load(f)

    #./data/ranking.csv 
    ranking = pd.read_csv(open(args.ranking, "r"),
                 sep=';',
                 engine='python')

    check_missing_images(original_dataset, ranking)


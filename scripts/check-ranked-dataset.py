import argparse
import json
import os
import pandas as pd

def check_missing_images(ranked_dataset, original_dataset):
    ranked_images=[]
    for i in range(0, len(ranked_dataset)):
        if ranked_dataset.loc[i, "case_name"] not in ranked_images:
            ranked_images.append(ranked_dataset.loc[i, "case_name"])
        if ranked_dataset.loc[i, "case_name_pos"] not in ranked_images:
            ranked_images.append(ranked_dataset.loc[i, "case_name_pos"])
        if ranked_dataset.loc[i, "case_name_neg"] not in ranked_images:
            ranked_images.append(ranked_dataset.loc[i, "case_name_neg"])

    i=0
    original_images=[]
    for i in range(0, len(original_dataset["images"])):
        if original_dataset["images"][i]["file_name"] in ranked_images:
            ranked_images.remove(original_dataset["images"][i]["file_name"])
    
    print("Missing images found: " + str(len(ranked_images)))
    print(ranked_images)


if __name__ == '__main__':
    '''script to check for missing image files
    expects a cli parameters:
        "--ranked_dataset" valued as csv for the ranked dataset
         "--original_dataset" valued by the json file path of the original dataset
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked_dataset", type=str, required=True)
    parser.add_argument("--original_dataset", type=str, required=True)
    args = parser.parse_args()
    #./data/oral_ranked_dataset.csv 
    ranked_dataset = pd.read_csv (open(args.ranked_dataset, "r"),
                 sep=';',
                 engine='python')
    
    #./data/dataset.json 
    with open(args.original_dataset, "r") as f:
        original_dataset = json.load(f)

    check_missing_images(ranked_dataset, original_dataset)


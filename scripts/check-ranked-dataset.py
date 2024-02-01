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

    #for row in ranked_images:
    #    ranking = ranking.drop(row)

    print("\nMissing anchor images found: " + str(len(anchors)) + "/" + str(total_size2))
    print(anchors)

    #for column in anchors:
    #    ranking = ranking.drop(column, axis=1)

def check_errors_and_ranking(ranking, ranking_path):
    print("\nErrors:")
    mox_rank_for_image={}
    to_drop=[]
    for index, row in ranking.iterrows():
        row_values=[]
        if index>=2:
            mox = -1
            analyzed_image=None
            for column, value in row.items():
                if column != 'id_casi' and column != 'TIPO DI ULCERA' and value != '-1':
                    row_values.append(int(value))
                    if mox < int(value):
                         mox = int(value)
                elif column == 'id_casi':
                    analyzed_image = value

            mox_rank_for_image[analyzed_image] = mox
            found_ranks=[]
            missing_ranks=[]
            duplicate_ranks=[]
            i=1
            while i != mox:
                if i in row_values and i not in found_ranks:
                    found_ranks.append(i)
                    row_values.remove(i)
                elif i in row_values and i in found_ranks:
                    duplicate_ranks.append(i)
                    i+=1
                elif i not in row_values and i not in found_ranks:
                    missing_ranks.append(i)
                    i+=1
                else:
                    i+=1
            if len(missing_ranks)!=0 or len(duplicate_ranks)!=0:
                print("Image:", analyzed_image,"missing ranks:",missing_ranks,"duplicate ranks:",duplicate_ranks)
                to_drop.append(index)

    while len(to_drop) != 0:
        index = to_drop.pop(0)
        ranking = ranking.drop(index)
    ranking.to_csv(ranking_path, sep=';', index=False, header=True)

    print("\nMax rank for image:")
    for key in mox_rank_for_image.keys():
        print(key + " " + str(mox_rank_for_image[key]))


if __name__ == '__main__':
    '''script to check for missing image files
    expects a cli parameters:
        "--original_dataset" valued by the json file path of the original dataset
        "--ranking" valued as csv for the ranking of images
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

    check_errors_and_ranking(ranking, args.ranking)


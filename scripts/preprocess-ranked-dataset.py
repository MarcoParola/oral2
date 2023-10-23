import pandas as pd
import argparse
import json

def preproces_ranked_dataset(dataset, correspondences):
    dataset = dataset.fillna(-1)
    combinations = []
    for index,row in dataset.iterrows():
        if index>=2:
            for col1 in dataset.columns:
                if col1 != 'id_casi' and col1 != 'TIPO DI ULCERA' and row[col1]!=-1 and col1!=row["id_casi"]:
                    for col2 in dataset.columns:
                        if col2 != 'id_casi' and col2 != 'TIPO DI ULCERA' and row[col2]!=-1:
                            ids=[]
                            if row["id_casi"] not in correspondences.keys():
                                ids.append("NotFound")
                            else:
                                ids.append(correspondences[row["id_casi"]])
                            
                            if col1 not in correspondences.keys():
                                ids.append("NotFound")
                            else:
                                ids.append(correspondences[col1])
                            
                            if col2 not in correspondences.keys():
                                ids.append("NotFound")
                            else:
                                ids.append(correspondences[col2])

                            if int(row[col1]) < int(row[col2]) and row["id_casi"] != col1 and row["id_casi"] != col2 and col2!=row["id_casi"]:
                                combinations.append([ids[0], row["id_casi"], row["TIPO DI ULCERA"], 
                                                    ids[1], col1, dataset.loc[0, col1], row[col1], 
                                                    ids[2], col2, dataset.loc[0, col2], row[col2]])

    new_dataset = pd.DataFrame(combinations, columns=['case_id', 'case_name', 'type', 'case_id_pos', 'case_name_pos', 'type_pos', 'rank_pos', 'case_id_neg', 'case_name_neg', 'type_neg', 'rank_neg'])
    new_dataset.to_csv('./data/oral_ranked_dataset.csv', sep=';', index=False, header=True)


if __name__ == '__main__':
    '''script to check for missing image files
    expects a cli parameters:
        "--ranking" valued as csv for the images' ranking
         "--original_dataset" valued by the json file path of the original dataset
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=str, required=True)
    parser.add_argument("--original_dataset", type=str, required=True)
    args = parser.parse_args()
    #./data/ranking.csv
    dataset = pd.read_csv (open(args.ranking, "r"),
                 sep=',',
                 engine='python')

    #"./data/dataset.json"
    with open(args.original_dataset, "r") as f:
        original_dataset = json.load(f)
          
        correspondences = dict()
        for image in original_dataset["images"]:
            correspondences[image["file_name"]] = image["id"]
    
    preproces_ranked_dataset(dataset, correspondences)

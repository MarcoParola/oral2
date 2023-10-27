import pandas as pd
import argparse
import json
import hydra
import numpy as np

@hydra.main(version_base=None, config_path="./../config", config_name="config")
def main(cfg):
    dataset = pd.read_csv(open(cfg.features_extractor.ranking, "r"),
                 sep=';',
                 engine='python')

    with open(cfg.dataset.original, "r") as f:
        original_dataset = json.load(f)
          
        correspondences = dict()
        for image in original_dataset["images"]:
            correspondences[image["file_name"]] = image["id"]

    not_considered_rows=[]
    not_considered_cols=[]
    dataset = dataset.fillna(-1)
    # delete empty rows/cols and rows with not allowed rank values
    for index, row in dataset.iterrows():
        if index>=2:
            safe = False
            for col_name, value in row.items():
                if not col_name.startswith('Unnamed'):
                    if col_name != "id_casi" and col_name != "TIPO DI ULCERA" and float(value)>=1.0:
                        safe = True
                else:
                    if col_name not in not_considered_cols:
                        not_considered_cols.append(col_name)
            if not safe:
                not_considered_rows.append(index)

    for i in range(0, len(not_considered_rows)):
        dataset=dataset.drop(not_considered_rows[i])
    
    for col in not_considered_cols:
        dataset=dataset.drop(col, axis=1)

    dataset = dataset.replace('0.8', '-1')

    dataset.to_csv(cfg.features_extractor.ranking, sep=';', index=False, header=True)

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
    new_dataset.to_csv(cfg.features_extractor.preprocessed_ranking, sep=';', index=False, header=True)


if __name__ == '__main__':
    main()

import torch
import numpy as np
import re
import pandas as pd
import json

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, features, ranking, img_dataset):
        features_dataset = pd.read_csv(open(features, "r"), sep=';', engine='python')

        self.triplets = pd.read_csv(open(dataset, "r"), sep=',', engine='python')
        triplets_ids=[]
        for i in range(0, len(self.triplets)):
            if self.triplets.loc[i, "case_id"] not in triplets_ids:
                triplets_ids.append(self.triplets.loc[i, "case_id"])
            if self.triplets.loc[i, "case_id_pos"] not in triplets_ids:
                triplets_ids.append(self.triplets.loc[i, "case_id_pos"])
            if self.triplets.loc[i, "case_id_neg"] not in triplets_ids:
                triplets_ids.append(self.triplets.loc[i, "case_id_neg"])

        i = 0
        while i < len(features_dataset):
            if features_dataset.loc[i, "image_id"] not in triplets_ids:
                features_dataset = features_dataset.drop(i)
            i+=1
        features_dataset.reset_index(inplace = True, drop = True)

        self.ids = list(features_dataset["image_id"])

        features = list(features_dataset["feature"])
        features=[np.array(eval(feature)) for feature in features]
        features = np.array(features)
        features = features.squeeze()
        features = torch.from_numpy(features)
        features = [feature.to(torch.float32) for feature in features]
        self.features = [feature.requires_grad_() for feature in features]

        dataset = json.load(open(img_dataset, "r"))
        image_names={}
        for image in dataset["images"]:
            if image["id"] in self.ids:
                image_names[image["file_name"]] = image["id"]

        self.ids_ranking={}
        image_names_keys = list(image_names.keys())

        ranking = pd.read_csv(open(ranking, "r"), sep=';', engine='python')
        for index, row in ranking.iterrows():
            row_values={} 
            if index>=2:
                mox = -1
                for column, value in row.items():
                    if (column != 'id_casi' and column != 'TIPO DI ULCERA' and value != '-1' 
                            and ranking.iloc[index, 0] in list(image_names.keys())):
                        row_values[int(value)]=column
                        if mox < int(value):
                            mox = int(value)
                if ranking.iloc[index, 0] in image_names_keys:
                    image_id = image_names[ranking.iloc[index, 0]]
                    j = 0
                    self.ids_ranking[image_id]=[]
                    for i in range(1, mox+1):
                        if row_values[i] in image_names:
                            id_rank = image_names[row_values[i]]
                            self.ids_ranking[image_id].append(id_rank)
                            j+=1


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        image_id = self.triplets.iloc[idx]["case_id"]
        index = self.ids.index(image_id)
        anchor = self.features[index]
        image_id = self.triplets.iloc[idx]["case_id_pos"]
        index = self.ids.index(image_id)
        positive = self.features[index]
        image_id = self.triplets.iloc[idx]["case_id_neg"]
        index = self.ids.index(image_id)
        negative = self.features[index]

        return anchor, positive, negative

    def get_ids_ranking(self):
        return self.ids_ranking

    def get_features_dataset(self):
        return self.ids, self.features


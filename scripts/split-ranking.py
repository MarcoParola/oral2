import argparse
import json
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import math


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--directory", type=str, default='')
parser.add_argument("--train_perc", type=float, default=0.7)
parser.add_argument("--val_perc", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

csv.field_size_limit(100000 * 10)

dataset = pd.read_csv(open(args.dataset, "r"),
            sep=';',
            engine='python')


found=[]
for index, row in dataset.iterrows():
        if index>=2:
            for col_name, value in row.items():
                if col_name == "case_id":
                    if value not in found:
                        found.append(value)

train, remaining = train_test_split(found, train_size=args.train_perc, random_state=args.seed)
val, test = train_test_split(remaining, train_size=0.5, random_state=args.seed)


train_data= pd.read_csv(open(args.dataset, "r"),
            sep=';',
            engine='python')
val_data= pd.read_csv(open(args.dataset, "r"),
            sep=';',
            engine='python')
test_data= pd.read_csv(open(args.dataset, "r"),
            sep=';',
            engine='python')


for index, row in dataset.iterrows():
        if index>=2:
            for col_name, value in row.items():
                if col_name == "case_id":
                    if value in test or value in val:
                        train_data = train_data.drop(index)
                    if value in train or value in test:
                        val_data = val_data.drop(index)
                    if value in train or value in val:
                        test_data = test_data.drop(index)

test_data = test_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

train_data.to_csv('./data/new_triplet/train.csv', index=False, sep=';')
val_data.to_csv('./data/new_triplet/val.csv', index=False, sep=';')
test_data.to_csv('./data/new_triplet/test.csv', index=False, sep=';')

print("OK!")



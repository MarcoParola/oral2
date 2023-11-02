import argparse
import json
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--train_perc", type=float, default=0.7)
parser.add_argument("--val_perc", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ranked_dataset = pd.read_csv(open(args.dataset, "r"),
                 sep=';',
                 engine='python')

train_data, remaining_data = train_test_split(ranked_dataset, train_size=args.train_perc, random_state=args.seed)
valid_data, test_data = train_test_split(remaining_data, train_size=0.5, random_state=args.seed)

train_data.to_csv('./data/triplet/train.csv', index=False)
valid_data.to_csv('./data/triplet/val.csv', index=False)
test_data.to_csv('./data/triplet/test.csv', index=False)
print("OK!")



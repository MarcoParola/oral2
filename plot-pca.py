import argparse
import torch
import torchvision
from tqdm import trange
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import re

parser = argparse.ArgumentParser()
parser.add_argument("--features_dataset", type=str, required=True)
args = parser.parse_args()
#./data/features_dataset.csv
features_dataset = pd.read_csv(open(args.features_dataset, "r"), sep=';', engine='python')

features = list(features_dataset["feature"])
features=[np.array(eval(feature)) for feature in features]
features = np.array(features)
features = features.squeeze()

pca = PCA(n_components=3)
pca.fit(features)
features = pca.transform(features)

labels = list(features_dataset["type"])
size_mapping = {0: 6, 1: 12, 2: 18}
sizes = [size_mapping[label] for label in labels]

plt.scatter(features[:, 0], features[:, 1], c=labels, s=sizes, alpha=.7)
plt.savefig("./outputs/img/pca1.png")
plt.scatter(features[:, 0], features[:, 2], c=labels, s=sizes, alpha=.7)
plt.savefig("./outputs/img/pca2.png")
plt.scatter(features[:, 1], features[:, 2], c=labels, s=sizes, alpha=.7)
plt.savefig("./outputs/img/pca3.png")
import argparse
import torch
import torchvision
from tqdm import trange
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.classifier import OralClassifierModule
from src.dataset import OralClassificationDataset

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)

args = parser.parse_args()

model = OralClassifierModule.load_from_checkpoint(args.model).eval()
data = OralClassificationDataset(args.dataset)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])

model.model.fc = torch.nn.Identity()

features = []
labels = []
for i in trange(len(data)):
    image, label = data[i]
    image = transform(image)
    out = model(image.unsqueeze(0))[0]

    features.append(out.detach().numpy())
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

pca = PCA(n_components=2)
pca.fit(features)

features = pca.transform(features)

plt.scatter(features[:, 0], features[:, 1], c=labels, s=4)
plt.savefig("pca.png")
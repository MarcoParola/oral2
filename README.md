# **Oral2**
[![license](https://img.shields.io/github/license/MarcoParola/oral2?style=plastic)]()
[![size](https://img.shields.io/github/languages/code-size/MarcoParola/oral2?style=plastic)]()

This github repo is to publicly release the code of oral2.

## Install

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```
python -m venv env
env/Scripts/activate
python -m pip install -r requirements.txt
mkdir data
```
Then you can download the oral coco-dataset (both images and json file) from TODO-put-link. Copy them into `data` folder and unzip the file `oral1.zip`.

## Usage
Regarding the usage of this repo, in order to reproduce the experiments, we organize the workflow in two part: (i) data preparation and (ii) deep learning experiments.

### Data preparation
Network for classification:
Due to the possibility of errors in the dataset, such as missing images, run the check-dataset.py script to detect such errors. Returns the elements to be removed from the json file (this can be done manually or via a script).
```
python -m scripts.check-dataset --dataset data\coco_dataset.json
```
In this work, the dataset was annotated with more labels than necessary. Some are grouped under more general labels. To aggregate all the labels of the three diseases studied in this work, the following script is executed. In addition, we split the dataset with the holdout method.
```
python -m scripts.simplify-dataset --folder data
python -m scripts.split-dataset --folder data
```

### Data visualization

You can use the `dataset-stats.py`   script to print the class occurrences for each dataset.
```
python -m scripts.dataset-stats --dataset data\dataset.json # entire dataset
python -m scripts.dataset-stats --dataset data\train.json # training set
python -m scripts.dataset-stats --dataset data\test.json # test set
```

Use the following command to visualize the dataset bbox distribution: 
```
python scripts\plot-distribution.py --dataset data\dataset.json
```

### DL experiments

## Classification model
```
python classifier-train.py
```

## Projection model
```
python triplet-train.py
```

# Example of a simple pipeline

- ```python classifier-train.py```
- Set the `features_extractor.checkpoint_path` in the projection configuration as the model for feature extraction.
- Set the `features_extractor.classifier=True` in the projection configuration in order to activate the feature extractor for the classifer features.
- ```python scripts.features-extractor```
- ```python plot-pca.py```
- Set the `triplet.projection=False` in the projection configuration in order to activate the ranking process without the triplet net.
- ```python rank-projection```
- ```python triplet-train.py```
- Set the `triplet.checkpoint_path` in the projection configuration as the projection model to use.
- Set the `triplet.projection=True` in the projection configuration in order to activate the ranking process.
- ```python rank-projection.py```
- Set the `features_extractor.classifier=False` in the projection configuration in order to activate the feature extractor for the projected features.
- ```python scripts.features-extractor```
- ```python plot-pca.py```



```
python -m tensorboard.main --logdir=logs
```

Esempio pca

```
python plot-pca.py --model resnet50.pth  --dataset data/train.json
```

Ho aggiunto le metriche per il calcolo della spiegabilità che valutano la simiraità tra due ranking.
Si usano due metriche `Spearman Footrule` e `Kendall Tau`. 
Andrà usato più o meno così:

```
from src.utils import convert_arrays_to_integers
from src.metrics import spearman_footrule_distance, kendall_tau_distance

vector1 = ['a','b','c','d', 'e']
vector2 = ['e','c','d','b', 'a']

print("vector1:", vector1)
print("vector2:", vector2)

vector1, vector2 = convert_arrays_to_integers(vector1, vector2)

print("ranked vector1:", vector1)
print("ranked vector2:", vector2)
print("Spearman distance:", spearman_footrule_distance(vector1, vector2))
print("Kendall distance:", kendall_tau_distance(vector1, vector2))
```
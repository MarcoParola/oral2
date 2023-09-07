# **Oral2**
This github repo is to publicly release the code of oral2.

## Install

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```
python -m venv env
env/Scripts/activate
python -m pip install -r requirements.txt
```

## Usage
Regarding the usage of this repo, in order to reproduce the experiments, we organize the workflow in two part: (i) data preparation and (ii) deep learning experiments.

### Data preparation

Data is composed of ... TODO

Il dataset te lo mando io quando ne avrai bisogno, quello di Marco non va bene perchè ci sono degli errori. Ti mando poi io la versione sistemata e già splittata

### DL experiments

Esempio train

```
python train.py log.tensorboard=True train.save_path=resnet50.pth
```

Esempio pca

```
python plot-pca.py --model resnet50.pth  --dataset dataset/oral/test.json
```



### Installazione

Crea virtualenv (poi anche usare `python -m venv`)

```
virtualenv --python=python3.11 env
. ./env/bin/activate
```

Installa requirements

```
pip install -r requirements.txt
```

Io ho usato `Python 3.11`.
Per la gestion del venv puoi anche usare conda come ti torna comodo

### Dataset

Il dataset te lo mando io quando ne avrai bisogno, quello di Marco non va bene perchè ci sono degli errori. Ti mando poi io la versione sistemata e già splittata

### Esempi

Esempio train

```
python train.py log.tensorboard=True train.save_path=resnet50.pth
```

Esempio pca

```
python plot-pca.py --model resnet50.pth  --dataset dataset/oral/test.json
```

### Risorse

- Studia come funzionano i [virtualenv](https://docs.python.org/3/library/venv.html) su python
- Per la configurazione studia [hydra](https://hydra.cc/docs/intro/)
- Per il training ho usato [pytorch-lightning](https://www.pytorchlightning.ai/index.html)
- Per i modelli [torchvision](https://pytorch.org/vision/stable/index.html)

Studia bene i framework e poi metti le mani sul codice


### Obiettivi

I tuoi obiettivi sono

- [x] Aggiungere metriche di classificazione 
- [ ] Sistemare il codice 
- [ ] Addestrare vari modelli sul problema di classificazione
- [ ] Train AutoEncoder (opzionale)
- [ ] crivere il codice per addestrare il layer di proiezione dallo spazio delle feature di classificazione a quello del medico esperto. Chiedi a [@MarcoParola](https://github.com/MarcoParola) il dataset
- [ ] Implementare un classificatore knn (con scikit) e vedere se la classificazione knn funziona meglio usando le feature della classificazione o le feature proiettate
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
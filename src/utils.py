import os
import re
import hydra
import torchvision
import numpy as np  
import pandas as pd
import seaborn as sn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.multiclass import unique_labels
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
import sys
sys.path.append('src')
from metrics import *
import cv2
import albumentations as A
import albumentations.pytorch as AP
import json


def get_loggers(cfg):
    """Returns a list of loggers
    cfg: hydra config
    """
    loggers = list()
    if cfg.log.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)
    
    if cfg.log.tensorboard:
        from pytorch_lightning.loggers import TensorBoardLogger
        tensorboard_logger = TensorBoardLogger(cfg.log.path , name=cfg.log.dir)
        loggers.append(tensorboard_logger)

    return loggers


def get_early_stopping(cfg):
    """Returns an EarlyStopping callback
    cfg: hydra config
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=25,
    )
    return early_stopping_callback


def get_transformations(cfg):
    """Returns the transformations for the dataset
    cfg: hydra config
    """
    img_tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
        torchvision.transforms.CenterCrop(cfg.dataset.resize),
        torchvision.transforms.ToTensor(),
    ])
    val_img_tranform, test_img_tranform = None, None

    train_img_tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
        torchvision.transforms.CenterCrop(cfg.dataset.resize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(degrees=45),
        #torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0),
    ])
    return train_img_tranform, val_img_tranform, test_img_tranform, img_tranform


'''def get_transformations(cfg):
    """Returns the transformations for the dataset
    cfg: hydra config
    """
    img_tranform = A.Compose([
        A.CLAHE(clip_limit=2.0, always_apply=True),
        A.Resize(cfg.dataset.resize, cfg.dataset.resize),
        A.CenterCrop(cfg.dataset.resize, cfg.dataset.resize),
        #A.ToFloat(),
        AP.transforms.ToTensorV2()
    ])
    val_img_tranform, test_img_tranform = None, None

    train_img_tranform = A.Compose([
        A.CLAHE(clip_limit=2.0, always_apply=True),
        A.ShiftScaleRotate(shift_limit=0.03, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.85), 
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, 
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, 
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
        )], p=0.2),

        A.OneOf([
            A.GaussNoise(var_limit=(0.0001, 0.004), p=0.7),
            A.Blur(blur_limit=3, p=0.3)
        ], p=0.5),      
        A.Flip(p=0.5),
        A.Resize(cfg.dataset.resize, cfg.dataset.resize),
        A.CenterCrop(cfg.dataset.resize, cfg.dataset.resize),
        #A.ToFloat(),
        AP.transforms.ToTensorV2()
    ])

    return train_img_tranform, val_img_tranform, test_img_tranform, img_tranform
'''

def log_report(actual, predicted, classes, log_dir):
    """Logs the confusion matrix to tensorboard
    actual: ground truth
    predicted: predictions
    classes: list of classes
    log_dir: path to the log directory
    """
    writer = SummaryWriter(log_dir=log_dir)
    cf_matrix = confusion_matrix(actual, predicted)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None], 
        index=[i for i in classes],
        columns=[i for i in classes])
    plt.figure(figsize=(5, 4))
    img = sn.heatmap(df_cm, annot=True, cmap="Greens").get_figure()
    writer.add_figure("Confusion matrix", img, 0)

    # log metrics on tensorboard
    writer.add_scalar('Accuracy', accuracy_score(actual, predicted))
    writer.add_scalar('recall', recall_score(actual, predicted, average='micro'))
    writer.add_scalar('precision', precision_score(actual, predicted, average='micro'))
    writer.add_scalar('f1', f1_score(actual, predicted, average='micro'))
    writer.close()



def get_last_version(path):
    """Return the last version of the folder in path
    path: path to the folder containing the versions
    """
    folders = os.listdir(path)
    # get the folders starting with 'version_'
    folders = [f for f in folders if re.match(r'version_[0-9]+', f)]
    # get the last folder with the highest number
    last_folder = max(folders, key=lambda f: int(f.split('_')[1]))
    return last_folder  

def log_compound_metrics(actual, predicted, accuracy, projected, log_dir):
    """Logs metrics to tensorboard
    actual: ground truth
    predicted: predictions
    accuracy: accuracy
    projected: a simple flag
    log_dir: path to the log directory
    """
    assert len(actual) == len(predicted)

    d1s=[]
    d2s=[]
    d3s=[]
    compounds=[]

    d3s_5=[]
    
    d3s_10=[]

    writer = SummaryWriter(log_dir=log_dir)
    for key in actual.keys():
        gt = actual[key]
        predict = predicted[key]

        d3 = jaccard_mod(gt, predict)
        d3_5 = jaccard_mod(gt[:5], predict[:5])
        d3_10 = jaccard_mod(gt[:10], predict[:10])

        to_del = []
        for i in range(0, len(predict)):
            if predict[i] not in gt:
                to_del.append(predict[i])

        for element in to_del:
            predict.remove(element)

        rank={}
        for i in range(0, len(gt)):
            rank[gt[i]] = i+1
            gt[i] = i+1

        for i in range(0, len(predict)):
            predict[i] = rank[predict[i]]

        d1 = spearman_footrule_distance(gt,predict)
        d2 = kendall_tau_distance(gt,predict)
        
        value = 0.3*d1 + 0.4*d2 + 0.3*d3
        
        d1s.append(d1)
        d2s.append(d2)
        d3s.append(d3)
        compounds.append(value)  

        d3s_5.append(d3_5)

        d3s_10.append(d3_10)    

    if projected:
        writer.add_scalar('mean compound distance projected', sum(compounds) / len(compounds))
        writer.add_scalar('mean spearman footrule distance projected', sum(d1s) / len(d1s))
        writer.add_scalar('mean kendall tau distance projected', sum(d2s) / len(d2s))
        writer.add_scalar('mean jaccard distance projected', sum(d3s) / len(d3s))
        writer.add_scalar('mean jaccard distance @ 5 projected', sum(d3s_5) / len(d3s_5))
        writer.add_scalar('mean jaccard distance @ 10 projected', sum(d3s_10) / len(d3s_10))
        writer.add_scalar('knn classification accuracy projected', accuracy)

        print('mean compound distance projected', sum(compounds) / len(compounds))
        print('mean spearman footrule distance projected', sum(d1s) / len(d1s))
        print('mean kendall tau distance projected', sum(d2s) / len(d2s))
        print('mean jaccard distance projected', sum(d3s) / len(d3s))
        print('mean jaccard distance @ 5 projected', sum(d3s_5) / len(d3s_5))
        print('mean jaccard distance @ 10 projected', sum(d3s_10) / len(d3s_10))
        print('knn classification accuracy projected', accuracy)
    else:
        writer.add_scalar('mean compound distance NOT projected', sum(compounds) / len(compounds))
        writer.add_scalar('mean spearman footrule distance NOT projected', sum(d1s) / len(d1s))
        writer.add_scalar('mean kendall tau distance NOT projected', sum(d2s) / len(d2s))
        writer.add_scalar('mean jaccard distance NOT projected', sum(d3s) / len(d3s))
        writer.add_scalar('mean jaccard distance @ 5 NOT projected', sum(d3s_5) / len(d3s_5))
        writer.add_scalar('mean jaccard distance @ 10 NOT projected', sum(d3s_10) / len(d3s_10))
        writer.add_scalar('knn classification accuracy NOT projected', accuracy)

        print('mean compound distance NOT projected', sum(compounds) / len(compounds))
        print('mean spearman footrule distance NOT projected', sum(d1s) / len(d1s))
        print('mean kendall tau distance NOT projected', sum(d2s) / len(d2s))
        print('mean jaccard distance NOT projected', sum(d3s) / len(d3s))
        print('mean jaccard distance @ 5 NOT projected', sum(d3s_5) / len(d3s_5))
        print('mean jaccard distance @ 10 NOT projected', sum(d3s_10) / len(d3s_10))
        print('knn classification accuracy NOT projected', accuracy)

    writer.close()

def get_k():
    """Returns the number of references
    """
    dataset = pd.read_csv(open("./data/ranking.csv", "r"), sep=';', engine='python')
    return dataset.shape[1]-2

def images_comparison(original, reconstructed, n):
    """Creates and save a comparison image between the original and autoencoder recostructed image
    original: original image
    reconstructed: reconstructed image
    n: result index
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    axes[0].imshow(original.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot reconstructed image
    axes[1].imshow(reconstructed.permute(1, 2, 0))
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    plt.savefig("./outputs/img/compare_"+str(n)+".png")
    plt.close()

def get_knn_classification(features_reference, lbls_reference, features_test, lbls_test, n_neighbors = 5):   
    """Compute and return the accuracy with knn
    features_reference: reference features 
    lbls_reference: labels of the reference features
    features_test: test features 
    lbls_test: labels of the test features
    n_neighbors: nearest neighbour 
    """ 
    distance_metric = 'euclidean'
    #distance_metric = 'cosine'
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=distance_metric)
    knn.fit(features_reference, lbls_reference)

    predicted_labels=[]
    for feature in features_test:
        predicted_labels.append(knn.predict([feature]))

    accuracy = accuracy_score(lbls_test, predicted_labels)

    return accuracy

def get_knn(ids, features_reference, features_test, n_neighbors = 20):    
    """Find the ids of the knn
    features_reference: reference features 
    features_test: test features 
    n_neighbors: nearest neighbour 
    """ 
    distance_metric = 'euclidean'
    #distance_metric = 'cosine'

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=distance_metric)
    knn.fit(features_reference)

    test_knn=[]
    for feature in features_test:
        distances, indices = knn.kneighbors([feature])
        test_knn.append([ids[i] for i in indices[0]])

    return test_knn
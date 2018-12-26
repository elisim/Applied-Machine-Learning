import pandas as pd
import numpy as np
import os
from copy import copy

TRAIN_PATH = 'input/saftey_efficay_myopiaTrain.csv'
TEST_PATH = 'input/saftey_efficay_myopiaTest.csv'
ROKAH_PATH = 'input/saftey_efficay_myopiaSample.csv'

def _pred2rokah_format(pred):
    """
    pred: classifier's predictions
    return: dataframe contains pred data in rokah format
    """
    ids = np.arange(1, pred.shape[0]+1)
    df = pd.DataFrame({"Id" : ids, "Class" : pred})
    return df

def to_file(pred, fname, to_kaggle=False, msg=""):
    df = _pred2rokah_format(pred)
    df.to_csv("out/{}.csv".format(fname), index=False)
    if to_kaggle:
        os.system('kaggle competitions submit -c bgu2018 -f out/{0}.csv -m "{1}"'.format(fname, msg))

def load_data():
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    return train,test

def remove_columns(train, test, columns_to_remove):
    train = train.drop(columns=columns_to_remove)
    test = test.drop(columns=columns_to_remove)
    return train, test

def preprocess_data(train, test, pipeline):
    """
    pipleline - pipeline to fit_transform the concat dummies before spliting to train & test
    return: train & train dummies after pipeline fit_transform
    """
    n_train = train.shape[0]
    concat = pd.concat(objs=[train, test], axis=0, sort=False)
    concat_dummies = pd.get_dummies(concat)
    concat_dummies = pipeline.fit_transform(concat_dummies)
    train_dummies = copy(concat_dummies[:n_train])
    test_dummies = copy(concat_dummies[n_train:])
    return train_dummies, test_dummies



import pandas as pd
import numpy as np
from copy import copy
from sklearn import preprocessing
from datasets.dataset import Dataset

TRAIN_PATH = 'data/abalone/abalone.data'

class AbaloneDataset(Dataset):
    
    def __init__(self):
        self._raw_train_data = pd.read_csv(TRAIN_PATH, header=None)
        
    def get_classes(self):
        return list(range(1,30))
    
    def get_train_and_test_data(self):
        # Package data into a dictionary
        return {
          'X_train': X_train_dummies, 'y_train': y_train_dummies,
          'X_test': X_test_dummies, 'y_test': y_test_dummies
        }

    @property
    def shape(self):
        return concat.shape
    
    def _concat_train_test(self):
        return pd.concat(objs=[self._raw_train_data, self._raw_test_data], axis=0)
    
    def _get_train_test_dummies(self):
        n_train = len(self._raw_train_data)
        concat = self._concat_train_test()
        concat_dummies = pd.get_dummies(concat)
        train_data_dummies = copy(concat_dummies[:n_train])
        test_data_dummies = copy(concat_dummies[n_train:])
        
        # drop the classes columns. after one-hot encoding, number of classes' columns will be the number of the classes
        n_features = train_data_dummies.shape[1]
        classes_cols_start = n_features - len(self.get_classes()) 
        X_train = train_data_dummies.iloc[:, :classes_cols_start]
        X_test = test_data_dummies.iloc[:, :classes_cols_start]     
        return X_train, X_test
    
    def _normalize_fnlwgt(self, data):
        """
        data - train or test data.
        fnlwgt (final weight) is the third column, and a normalization is needed to avoid overflow. 
        We decided to use mean normalization.
        """
        fnlwgt = data.iloc[:,2]
        fnlwgt = (fnlwgt - fnlwgt.mean())/(fnlwgt.std())
        data.iloc[:,2] = fnlwgt
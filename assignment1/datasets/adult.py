import pandas as pd
import numpy as np
from copy import copy
from sklearn import preprocessing
from datasets.dataset import Dataset

class AdultDataset(Dataset):
    
    def __init__(self, train_path, test_path):
        self._raw_train_data = pd.read_csv(train_path, header=None)
        self._raw_test_data = pd.read_csv(test_path, header=None)
        assert self._raw_train_data.shape[1] == self._raw_test_data.shape[1], "The number of features in train & test is inequal"

    
    def get_classes(self):
        return ['<=50k', '>50k']
    
    def get_train_and_test_data(self):
        le = preprocessing.LabelEncoder()
        y_train_dummies = le.fit_transform(self._raw_train_data.iloc[:, -1])
        y_test_dummies = le.fit_transform(self._raw_test_data.iloc[:, -1])
        X_train_dummies, X_test_dummies = self._get_train_test_dummies()
        # Package data into a dictionary
        return {
          'X_train': X_train_dummies, 'y_train': y_train_dummies,
          'X_test': X_test_dummies, 'y_test': y_test_dummies
        }

    @property
    def shape(self):
        concat = self._concat_train_test()
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
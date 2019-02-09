import pandas as pd
import numpy as np
from copy import copy
from sklearn import preprocessing
from datasets.dataset import Dataset

TRAIN_PATH = 'data/adult/adult.data'
TEST_PATH = 'data/adult/adult.test'

class AdultDataset(Dataset):
    
    def __init__(self):
        self._raw_data = pd.read_csv(TRAIN_PATH, header=None)
        self._normalize_fnlwgt(self._raw_data)
        
    def get_classes(self):
        return ['<=50k', '>50k']
    
    def get_train_and_test_data(self):
        le = preprocessing.LabelEncoder()
        y_train_dummies = le.fit_transform(self._raw_train_data.iloc[:, -1])
        X_train_dummies, X_test_dummies = self._get_train_test_dummies()
        # Package data into a dictionary
        return {
          'X_train': X_train_dummies, 'y_train': y_train_dummies,
          'X_test': X_test_dummies, 'y_test': y_test_dummies
        }

    @property
    def shape(self):
        return self._raw_data .shape
    
    
#     def get_data():
#         """
#         Returns the data after get_dummies, and labels after encoding
#         """
#         y = dummies.iloc[: ,[-1]] # last column after get dummies
#         dummies = pd.get_dummies(self._raw_train_data)
#         dummies.iloc[: ,[-4]] = self._le.fit_transform(y).values
#         return dummies
    
    
    def _normalize_fnlwgt(self, data):
        """
        data - train or test data.
        fnlwgt (final weight) is the third column, and a normalization is needed to avoid overflow. 
        We decided to use mean normalization.
        """
        fnlwgt = data.iloc[:,2]
        fnlwgt = (fnlwgt - fnlwgt.mean())/(fnlwgt.std())
        data.iloc[:,2] = fnlwgt
        
    
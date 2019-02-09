from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn import preprocessing

class Dataset(ABC):
    
    @property
    @abstractmethod
    def shape(self):
        """
        The shape of the dataset, meaning [n_samples, n_features]
        """
        pass
    
    @abstractmethod
    def get_classes(self):
        """
        list of the classes 
        """
        pass
        
    @abstractmethod
    def get_train_and_test_data(self):
        """
        Returns a dict with the keys: X_train, y_train, X_test, y_test
        """
        pass
    
    
    def get_X_y(self):
        """
        Returns the data after get_dummies, and labels after encoding
        """
        X, y = self._concat(self.get_train_and_test_data())
        X, y = self._remove_samples(X,y,10) # delete rows with less than 10 samples with class
        y = preprocessing.LabelEncoder().fit_transform(y)
        return X, y
        
    
    def _concat(self, data):
        """
        concatenate train data with test data
        """
        X = pd.concat([data['X_train'], data['X_test']], axis=0)
        y = pd.concat([data['y_train'], data['y_test']], axis=0)
        if (len(y.shape) == 2): # y.shape is [rows, 1], otherwise is (rows,)
            return X, y.iloc[:, 0]
        return X,y 
    
    def _remove_samples(self, X, y, n_samples):
        bad_labels = np.array([k for k,v in y.value_counts().items() if v < n_samples])
        bad_label_idx = [i for i,x in enumerate(y.values) if x in bad_labels]
        X = X.drop(X.index[bad_label_idx])
        y = y.drop(y.index[bad_label_idx])
        return X,y
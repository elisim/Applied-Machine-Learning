from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datasets.dataset import Dataset

TRAIN_PATH = 'data/lymphography/lymphography.data'
n_features = 18

class LymphographyDataset(Dataset):
    
    def __init__(self):
        self._raw_train_data = pd.read_csv(TRAIN_PATH, names=["c" + str(i) for i in range(n_features)] + ["target"])
        
    def get_classes(self):
        return [str(age) for age in range(1,30)]
    
    def get_train_and_test_data(self):
        X = self._raw_train_data 
        y = X.target
        X = X.drop(columns=['target'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        # Package data into a dictionary
        return {
          'X_train': X_train, 'y_train': y_train,
          'X_test': X_test, 'y_test': y_test
        }

    @property
    def shape(self):
        return self._raw_train_data.shape
    

        

    
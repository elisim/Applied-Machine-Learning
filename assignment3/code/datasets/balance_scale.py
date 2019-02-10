from sklearn.model_selection import train_test_split
import pandas as pd
from datasets.dataset import Dataset

TRAIN_PATH = 'data/balance-scale/balance-scale.data'
n_features = 4

class BalanceScaleDataset(Dataset):
    
    def __init__(self):
        self._raw_train_data = pd.read_csv(TRAIN_PATH, names=["c" + str(i) for i in range(n_features)] + ["target"])
        self.name = 'balance_scale'
        
    def get_classes(self):
        return [str(class_index) for class_index in range(1,6)]
    
    def get_train_and_test_data(self):
        X = self._raw_train_data
        y = X.iloc[: ,[-1]] # last column 
        X = X.drop(y, axis=1)
        X_dummies = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.25, shuffle=True)
        # Package data into a dictionary
        return {
          'X_train': X_train, 'y_train': y_train,
          'X_test': X_test, 'y_test': y_test
        }

    @property
    def shape(self):
        return self._raw_train_data.shape
    
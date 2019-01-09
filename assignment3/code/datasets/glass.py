from sklearn.model_selection import train_test_split
import pandas as pd
from datasets.dataset import Dataset

TRAIN_PATH = 'data/glass/glass.data'

class GlassDataset(Dataset):
    
    def __init__(self):
        self._raw_train_data = pd.read_csv(TRAIN_PATH, header=None)
        
    def get_classes(self):
        return [str(glass_type) for glass_type in range(1,8)]
                
    
    def get_train_and_test_data(self):
        X = self._raw_train_data
        X = X.drop(X.iloc[:, [0]], inplace=False)
        y = X.iloc[:, [-1]]
        X = X.drop(y, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        # Package data into a dictionary
        return {
          'X_train': X_train, 'y_train': y_train,
          'X_test': X_test, 'y_test': y_test
        }

    @property
    def shape(self):
        return self._raw_train_data.shape
    
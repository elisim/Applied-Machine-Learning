from sklearn.model_selection import train_test_split
import pandas as pd
from datasets.dataset import Dataset
from sklearn import preprocessing

TRAIN_PATH = 'data/car/car.data'

class CarDataset(Dataset):
    
    def __init__(self):
        self._raw_train_data = pd.read_csv(TRAIN_PATH, header=None)
        
    def get_classes(self):
        return ['unacc', 'acc', 'good', 'v-good']
                
    
    def get_train_and_test_data(self):
        X_dummies, y_dummies = self._to_dummies()
        X_train, X_test, y_train, y_test = train_test_split(X_dummies, y_dummies, test_size=0.25, shuffle=True)
        # Package data into a dictionary
        return {
          'X_train': X_train, 'y_train': y_train,
          'X_test': X_test, 'y_test': y_test
        }

    @property
    def shape(self):
        return self._raw_train_data.shape
    
    def _to_dummies(self):
        """
        use one hot encoding on the dataset.
        """
        le = preprocessing.LabelEncoder()
        X = self._raw_train_data
        y = X.iloc[:, [-1]]
        y_dummies =  le.fit_transform(y)
        X = X.drop(y, axis=1)
        return pd.get_dummies(X), y_dummies
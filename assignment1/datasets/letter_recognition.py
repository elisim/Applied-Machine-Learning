from sklearn.model_selection import train_test_split
import pandas as pd
from datasets.dataset import Dataset
from string import ascii_uppercase 
from sklearn import preprocessing

TRAIN_PATH = 'data/letter-recognition/letter-recognition.data'

class LetterRecognitionDataset(Dataset):
    
    def __init__(self):
        self._raw_train_data = pd.read_csv(TRAIN_PATH, header=None)
        
    def get_classes(self):
        return list(ascii_uppercase)
    
    def get_train_and_test_data(self):
        X_dummies, y_dummies = self._letters_to_dummes()
        X_train, X_test, y_train, y_test = train_test_split(X_dummies, y_dummies, test_size=0.25, shuffle=True)
        # Package data into a dictionary
        return {
          'X_train': X_train, 'y_train': y_train,
          'X_test': X_test, 'y_test': y_test
        }

    @property
    def shape(self):
        return self._raw_train_data.shape
    
    def _letters_to_dummes(self):
        """
        Replace the first column with dummies
        """
        le = preprocessing.LabelEncoder()
        X = self._raw_train_data
        y = X.iloc[:, [0]]
        y_dummies =  le.fit_transform(y)
        X.iloc[:, [0]] = y_dummies
        return X.drop(y, axis=1), y_dummies
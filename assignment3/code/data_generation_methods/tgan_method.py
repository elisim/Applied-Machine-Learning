import pandas as pd
import pandas.api.types as pd_types
import numpy as np

class TganDataGeneration():
    """
    Data generation using TGAN: TGAN: A Tabular Data Synthesizer (github.com/DAI-Lab/TGAN)
    """
    
    def __init__(self):
        self._synthetic = None
    
    def gen_data(self, X, art_factor):
        """
        X - pandas dataframe with header 
        art_factor - factor that determines number of artificial examples to generate.
                     By default, art_factor = 1
        the parameters doesn't has usage because synthetic already created
        """
        
        self._synthetic = pd.read_csv
        
        
        return X_art
                
    def label_data(self, X_art, y_probs):
        """
        X_art - artificially generated examples
        y_probs - class membership probabilities of instance
        return: lables such that the probability of selection is inversely proportional to the ensemble's predictions
        """
        ans = []

        return np.array(ans)

    def __repr__(self):
        return '<class \'{}\'>'.format(self.__class__.__name__)
        
        
        

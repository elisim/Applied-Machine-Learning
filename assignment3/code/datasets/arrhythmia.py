from sklearn.model_selection import train_test_split
import pandas as pd
from datasets.dataset import Dataset

TRAIN_PATH = 'data/arrhythmia/arrhythmia.data'
n_features = 279

class ArrhythmiaDataset(Dataset):
    
    def __init__(self):
        self._raw_train_data = pd.read_csv(TRAIN_PATH, names=["c" + str(i) for i in range(n_features)] + ["target"])

    def get_classes(self):
        """
        Class code :   Class   :               
        01             Normal				        
        02             Ischemic changes (Coronary Artery Disease)
        03             Old Anterior Myocardial Infarction        
        04             Old Inferior Myocardial Infarction        
        05             Sinus tachycardy			           
        06             Sinus bradycardy			          
        07             Ventricular Premature Contraction (PVC)
        08             Supraventricular Premature Contraction
        09             Left bundle branch block 		     
        10             Right bundle branch block		     
        11             1. degree AtrioVentricular block	     
        12             2. degree AV block		            
        13             3. degree AV block		            
        14             Left ventricule hypertrophy 	        
        15             Atrial Fibrillation or Flutter	    
        16             Others				          
        """
        return [str(class_index) for class_index in range(1,16)]
    
    def get_train_and_test_data(self):
        data_without_missing = self._raw_train_data.replace(to_replace='?', value=0, inplace=False)
        y = data_without_missing.iloc[: ,-1] # last column 
        X = data_without_missing.drop(columns=['target'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        # Package data into a dictionary
        return {
          'X_train': X_train, 'y_train': y_train,
          'X_test': X_test, 'y_test': y_test
        }

    @property
    def shape(self):
        return self._raw_train_data.shape
    
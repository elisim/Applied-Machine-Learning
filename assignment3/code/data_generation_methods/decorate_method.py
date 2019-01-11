import pandas as pd
import pandas.api.types as pd_types
import numpy as np

class DecorateDataGeneration():
    """
    numeric attribute: compute the mean and standard deviation from the training set and generate columns from the Gaussian distribution defined by these.
    nominal attribute: compute the probability of occurrence of each distinct column in its domain and generate columns based on this distribution. 
    """
    
    def __init__(self):
        pass
    
    def gen_data(self, X, art_factor):
        """
        X - pandas dataframe with header
        art_factor - factor that determines number of artificial examples to generate
        """
        X_art = pd.DataFrame()
        types = X.dtypes
        n_art_samples = int(art_factor*len(X)) # compute number of artficial samples
        
        for col_idx, (key, column) in enumerate(X.iteritems()):
            col_type = types[col_idx]
            if pd_types.is_numeric_dtype(col_type): # numeric
                art_data = np.random.normal(column.mean(), column.std(), n_art_samples)
            elif pd_types.is_object_dtype(col_type): # nominal
                art_data = self._gen_nominal_data(column)
            else:
                raise TypeError("Decorate can only handle numeric and nominal columns, but got '{}' type".format(col_type))
            X_art[key] = pd.Series(art_data)
        
        return X_art
                
    def label_data(self, X_art, y_probs):
        """
        X_art - artificially generated examples
        y_probs - class membership probabilities of instance
        return: lables such that the probability of selection is inversely proportional to the ensemble's predictions
        """
        ans = []
        for y_prob in y_probs:
            inv_probs = np.zeros(len(y_prob), dtype=float)
            for i, prob in enumerate(y_prob):
                if prob == 0:
                    inv_probs[i] = (2 ** 10)/len(y_prob)
                else:
                    inv_probs[i] = 1.0/prob
            inv_probs = inv_probs/np.sum(inv_probs)
            
            # Calculate cumulative probabilities
            stats = [None]*len(inv_probs)
            stats[0] = inv_probs[0]
            for i in range(1, len(inv_probs)):
                stats[i] = stats[i-1] + inv_probs[i]
            ans.append(self._select_index_probabilistically(stats))

        return np.array(ans)

    
    def _gen_nominal_data(self, column):
        nom_counts = dict(column.value_counts()).values() # counts of each nominal value
        nom_counts = np.array(list(nom_counts)) # to numpy array
        if (len(nom_counts) < 2):
            raise TypeError('Nominal attribute has less than two distinct values')
            
        ### Perform Laplace smoothing
        counts = nom_counts + 1
        counts = counts/np.sum(counts)
        
        # Calculate cumulative probabilities
        stats = [None]*(len(counts)-1)
        stats[0] = counts[0]
        for i in range(1, len(stats)):
            stats[i] = stats[i-1] + counts[i]
        
        return self._select_index_probabilistically(stats)
    
    def _select_index_probabilistically(self, stats):
        rnd = np.random.uniform()
        index = 0
        while (index < len(stats) and rnd > stats[index]):
            index += 1
        return index
    
    def __repr__(self):
        return '<class \'{}\'>'.format(self.__class__.__name__)
        
        
        

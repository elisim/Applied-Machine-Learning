from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from data_generation_methods import *

class DecorateClassifier(BaseEstimator, ClassifierMixin):  
    """
     DECORATE is a meta-learner for building diverse ensembles of classifiers by using specially constructed artificial training examples.
    """

    def __init__(self, 
                 base_estimator=DecisionTreeClassifier,  
                 gen_artificial_method=DecorateDataGeneration(),
                 n_estimators=15,
                 n_iter=50,
                 art_factor=1.0,
                 random_seed=1.0
                 ):
        """
        base_estimator - base learning algorithm (DecisionTreeClassifier as default)
        gen_artificial_method - class that specify how to generate the artificial examples (Decorate method as default)
        n_estimators - desired ensemble size (default 15)
        n_iter - maximum number of iterations to build an ensemble (default 50)
        art_factor - factor that determines number of artificial examples to generate (default 1.0)
        random_seed - random number seed (default 1)
        """
        self.base_estimator = base_estimator
        self.gen_artificial_method = gen_artificial_method
        self.n_estimators = n_estimators
        self.n_iter = n_iter
        self.art_factor = art_factor 
        self.random_seed = random_seed
        self.ensemble_ = [] # initialize ensemble
        
    def fit(self, X, y):
        trials = 1 # number of Decorate iterations 
        
        # initialize first estimator
        estimator = self.base_estimator()
        estimator.fit(X,y)
        self.ensemble_.append(estimator)

        # compute ensemble error
        ens_error = self._compute_error(X, y) 
        
        # repeat till desired ensemble size is reached OR the max number of iterations is exceeded 
        while len(self.ensemble_)<self.n_estimators and trials<self.n_iter:
            
            # generate artificial training examples
            X_art = self.gen_artificial_method.gen_data(X, self.art_factor)
            
            # label artificial examples
            y_art = self.gen_artificial_method.label_data(X_art, self.predict_proba(X_art))
            
            # add new artificial data
            X_concat, y_concat = self._concat(X, y, X_art, y_art)
            
            # build new estimator
            new_estimator = self.base_estimator()
            new_estimator.fit(X_concat, y_concat)
            
            # test if the new estimator should be added to the ensemble
            self.ensemble_.append(new_estimator)
            curr_ens_error = self._compute_error(X, y) 
            
            if curr_ens_error <= ens_error:
                 # if adding the new member did not increase the error
                ens_error = curr_ens_error
            else: 
                # reject the current classifier because it increased the ensemble error 
                self.ensemble_.pop()
                
            trials += 1
            
        return self

    def predict_proba(self, X):
        """
        sum over all estimators predictions, and divide by the size of the ensemble
        """
        ans = np.zeros((len(X), 2), dtype=float)
        for estimator in self.ensemble_:
            pred = estimator.predict_proba(X)
            ans += pred
        return ans / len(self.ensemble_)
    
    def predict(self, X):
        """
        the class will be the class with the maximum probability
        """
        pred_prob = self.predict_proba(X)
        return np.argmax(pred_prob, axis=1) 
         
    def _compute_error(self, X, y):
        """
        number of samples misclassified
        """
        y_pred = self.predict(X)
        num_wrong = np.sum(y_pred != y)
        return num_wrong
    
    def _concat(self, X, y, X_art, y_art):
        """
        concatenate data with artificial data
        """
        X_concat = np.vstack((X, X_art))
        y_concat = np.hstack((y, y_art))
        return X_concat, y_concat
    
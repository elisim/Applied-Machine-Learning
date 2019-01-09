from abc import ABC, abstractmethod
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
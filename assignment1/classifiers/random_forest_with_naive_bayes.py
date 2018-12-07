from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from collections import defaultdict

class NaiveBayesModelTree(DecisionTreeClassifier):
    """
     Decision tree classifier model with a naive bayes model in the leaves. 
    """
    def __init__(self,
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                class_weight=None,
                presort=False):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort)
        self.__leaves = {}

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        super().fit(X, y,
                    sample_weight=sample_weight,
                    check_input=check_input,
                    X_idx_sorted=X_idx_sorted)
        
        leaves = self.tree_.apply(X)
        leaf_to_instances = defagit aultdict(list)
        for instance_index, leaf in enumerate(leaves):
            leaf_to_instances[leaf].append(instance_index)
        # For each leaf, create SmartLeaf object which hold the naive bayes model trained on the instances that
        # reached this leaf and save it in the tree state.
        for leaf_index, instance_indexes in leaf_to_instances.items():
            self._leaves[leaf_index] = SmartLeaf(leaf_index, X[instance_indexes,], y[instance_indexes])
        return self
        
    def predict_proba():
        """
        Override the original predict_proba by simply calling the relevant naive bayes predict_proba
        """
        X = self._validate_X_predict(X, check_input)
         # Find the leaf each instance reach
        leaf_indexes = self.apply(X)
         # Create placeholder matrix for the result
        results = np.zeros(shape=(X.shape[0], self.n_classes_))
         # For each instance call naive bayes predict_proba of the matching leaf and insert to result.
        for instance_index, leaf_index in enumerate(leaf_indexes):
            results[instance_index] = self._leaves[leaf_index].model.predict_proba([X[instance_index]])
        return results
    

class NaiveBayesLeaf(object):
    """
    Holds a naive bayes model.
    """
    def __init__(self, leaf_id, X, y):
        self._leaf_id = leaf_id
        self.model = self._add_naive_bayes_model(X, y)
    def _add_naive_bayes_model(self, X, y):
        naive_bayes_model = GaussianNB()
        return naive_bayes_model.fit(X, y.ravel())



    
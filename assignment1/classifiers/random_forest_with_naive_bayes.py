from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from collections import defaultdict

class NaiveBayesModelTreeClassifier(DecisionTreeClassifier):
    """
    Decision tree that uses a Naive Bayes model in the leaves. 
    This tree will be used as a base_estimator for the Random Forest Classifier.
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
        self._leaves = {}

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        """
        Override fit to use a Naive Bayes model.
        """
        # Create the tree as usual
        super(DecisionTreeClassifier, self).fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)

        leaves_indices = self.tree_.apply(X) # get the index of the leaf that each sample is predicted as.
        leaf_to_instances = defaultdict(list) # dict with a leaf_index -> instances_indices mapping

        for instance_index, leaf_index in enumerate(leaves_indices):
            leaf_to_instances[leaf_index].append(instance_index)

        # For each leaf, create a NaiveBayesLeaf which holds a naive bayes model 
        # trained on the instances that reached this leaf.
        for leaf_index, instances_indices in leaf_to_instances.items():
            self._leaves[leaf_index] = NaiveBayesLeaf(leaf_index)
            self._leaves[leaf_index].fit(X[instances_indices,], y[instances_indices])

        return self

    def predict_proba(self, X, check_input=True):
        """
        Override the original predict_proba by simply calling the NB predict_proba
        """
        X = self._validate_X_predict(X, check_input)

        # Find the leaf each instance reach
        leaves_indices = self.apply(X)

        # Create placeholder matrix for the result
        n_samples = X.shape[0]
        results = np.zeros(shape=(n_samples, self.n_classes_))

        # For each instance call naive bayes predict_proba of the matching leaf and insert to result.
        for instance_index, leaf_index in enumerate(leaves_indices):
            bayes_result = self._leaves[leaf_index].model.predict_proba([X[instance_index]])
            model_classes = list(self._leaves[leaf_index].model.classes_)
            
            # handle the case where the leaf didn't see all the classes. 
            # In such case, predict zero for each class like that.
            result_with_all_classes = np.zeros(shape=self.n_classes_)
            for class_number in range(0, self.n_classes_):
                if class_number in model_classes:
                    result_with_all_classes[class_number] = bayes_result[0, model_classes.index(class_number)]
            results[instance_index] = result_with_all_classes

        return results


class RandomForestWithNaiveBayesLeavesClassifier(RandomForestClassifier):
    """
    A Random Forest model that uses NaiveBayesModelTreeClassifier
    """
    def __init__(self,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForestClassifier, self).__init__(
            base_estimator=NaiveBayesModelTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        
class NaiveBayesLeaf(object):
    """
    Leaf that holds a naive bayes model trained on the instances that got
    to it during training. 
    """
    def __init__(self, leaf_index):
        self.leaf_index = leaf_index
        self.model = None
    
    def fit(self, X, y):
        naive_bayes_model = GaussianNB()
        self.model = naive_bayes_model.fit(X,y)
    
    

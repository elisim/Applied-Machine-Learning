import time                                                
from sklearn.metrics import classification_report
            
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{0} took {1:.2f} sec'.format(method.__name__, te-ts))
        return result
    return timed


class Experiment(object):
    """
    model1 - standard ramdom forest 
    model2 - random forest with naive bayes in the leaves
    dataset - Dataset object from './datasets/'
    """
    def __init__(self, model1, model2, dataset):
        self._model1 = model1
        self._model2 = model2
        self._dataset = dataset
        self._y_test_pred = None
        self._data = None
       
    @timeit
    def run(self):
        self._data = self._dataset.get_train_and_test_data()
        self._model1.fit(self._data['X_train'], self._data['y_train'])
        self._model2.fit(self._data['X_train'], self._data['y_train'])
        self._y_test_pred = [self._model1.predict(self._data['X_test']), self._model2.predict(self._data['X_test'])]
        
    def get_results(self):
        if self._y_test_pred == None:
            raise ValueError("run the experiment first")
        return  {
            "model1": classification_report(self._data['y_test'], self._y_test_pred[0], target_names=self._dataset.get_classes()),
            "model2": classification_report(self._data['y_test'], self._y_test_pred[1], target_names=self._dataset.get_classes()),
        }
        
    def print_results(self):
        results = self.get_results()
        print("Dataset: {dataset}\nshape: {shape}\n{model1} results:\n\n{model1_res}\n{model2} results:\n\n{model2_res}".format(
                dataset=self._dataset.__class__.__name__,
                shape=self._dataset.shape,
                model1=self._model1.__class__.__name__,
                model1_res=results['model1'],
                model2=self._model2.__class__.__name__,
                model2_res=results['model2']))

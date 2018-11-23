import urllib.request
with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data") as url:
    dataset = url.read()
X = dataset[0:13]
y = dataset[14]

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=32561, n_features=15,n_informative=15, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X, y)

clf.predict([45, 'Local-gov', 119199, 'Assoc-acdm', 12, 'Divorced', 'Prof-specialty', 'Unmarried, White', 'Female', 0, 0, 48, 'United-States'])
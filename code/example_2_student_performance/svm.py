from explore_dataset import load_preproccessed_dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import pandas as pd

(X, y), (test_X, test_y) = load_preproccessed_dataset(test_split=0.1, include_grades=True)

stadardise = StandardScaler()
X = stadardise.fit_transform(X)

# Kaiser stopping
kaiser_pca = PCA(n_components=10)
x_kaiser = kaiser_pca.fit_transform(X)

# Scree test
scree_pca = PCA(n_components=3)
x_scree = scree_pca.fit_transform(X)

hyperparams = {
    'gamma': [2e-9, 2e-7, 2e-5, 2e-3, 2e-1, ],
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly',],
}

results = {}
svm = SVC()

clf_base = GridSearchCV(svm, param_grid=hyperparams, cv=10)
clf_base.fit(X, y)
results['Base'] = pd.DataFrame(clf_base.cv_results_)

clf_kaiser = GridSearchCV(svm, param_grid=hyperparams, cv=10)
clf_kaiser.fit(x_kaiser, y)
results['Kaiser'] = pd.DataFrame(clf_kaiser.cv_results_)

clf_scree = GridSearchCV(svm, param_grid=hyperparams, cv=10)
clf_scree.fit(x_scree, y)
results['Scree'] = pd.DataFrame(clf_scree.cv_results_)

for test, result in results.items():
    print(test)
    print(result[[*(f'param_{p}' for p in hyperparams.keys()), 'mean_fit_time', 'mean_test_score', 'std_test_score']], end='\n\n')

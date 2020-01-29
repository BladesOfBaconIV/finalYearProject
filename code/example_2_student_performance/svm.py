from explore_dataset import load_preproccessed_dataset

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

import pandas as pd

(X, y), (test_X, test_y) = load_preproccessed_dataset(test_split=0.1, include_grades=True)

stadardise = StandardScaler()
X = stadardise.fit_transform(X)

print(X.shape)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_reduced = model.transform(X)
test_X_reduced = model.transform(test_X)

print(X_reduced.shape)

hyperparams = {
    'gamma': [2e-9, 2e-7, 2e-5, 2e-3, 2e-1, ],
    'kernel': ['poly', 'rbf', 'sigmoid', ],
}

svm = SVC()
clf = GridSearchCV(svm, param_grid=hyperparams, cv=10)
clf.fit(X_reduced, y)
cv_results_df = pd.DataFrame(clf.cv_results_)

print(cv_results_df[[*(f'param_{p}' for p in hyperparams.keys()), 'mean_fit_time', 'mean_test_score', 'std_test_score']])

clf2 = GridSearchCV(svm, param_grid=hyperparams, cv=10)
clf2.fit(X, y)
cv_results_df_2 = pd.DataFrame(clf2.cv_results_)

print(cv_results_df_2[[*(f'param_{p}' for p in hyperparams.keys()), 'mean_fit_time', 'mean_test_score', 'std_test_score']])

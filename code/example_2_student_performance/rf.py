from explore_dataset import load_preproccessed_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd

(X, y), _ = load_preproccessed_dataset(test_split=0.0, include_grades=True)

hyperparams = {
    'n_estimators': [100, 500,],
    'max_depth': [3, None,],
    'min_samples_leaf': [1, 0.05,],
}

clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=hyperparams, cv=10)
clf.fit(X, y)

cv_results = pd.DataFrame(clf.cv_results_)
print(cv_results[[*(f'param_{p}' for p in hyperparams.keys()), 'mean_fit_time', 'mean_test_score', 'std_test_score']])

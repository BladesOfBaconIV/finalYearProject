from explore_dataset import load_preproccessed_dataset

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

(X, y), (test_X, test_y) = load_preproccessed_dataset(test_split=0.1, include_grades=True)

hyperparams = {
    'algorithm': ['SAMME', 'SAMME.R'],
    'n_estimators': [500],
}

num_param_combinations = np.product(list(map(len, hyperparams.values())))
total_mean_scores = np.zeros((num_param_combinations, 2))
best_classifiers = []
for i in range(20):
    print(f'Starting {i}')
    rf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2))
    clf = GridSearchCV(rf, param_grid=hyperparams, cv=10)
    clf.fit(X, y)
    cv_results_df = pd.DataFrame(clf.cv_results_)
    total_mean_scores += cv_results_df[['mean_test_score', 'std_test_score']]
    best_classifiers.append(clf.best_estimator_)
    print(f'Ending {i}\n')

total_mean_scores /= 20
params = cv_results_df[[f'param_{p}' for p in hyperparams.keys()]]
results = pd.concat([params, total_mean_scores], axis=1)
print(results)

best_classifiers_average_test = sum(clf.score(test_X, test_y) for clf in best_classifiers) / 20
print(f'\nBest classifier average test accuracy: {best_classifiers_average_test}')

from explore_dataset import load_preproccessed_dataset

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

(X, y), (test_X, test_y) = load_preproccessed_dataset(test_split=0.1, include_grades=True)

"""
Calculating the ccp_alpha that gives the best test score and prevents over-fitting
"""
dt = DecisionTreeClassifier(random_state=42)
path = dt.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X, y)
    clfs.append(clf)

train_scores = [clf.score(X, y) for clf in clfs]
test_scores = [clf.score(test_X, test_y) for clf in clfs]

best_alpha = ccp_alphas[np.argmax(test_scores)]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

hyperparams = {
    'max_depth': [3, 5, None],
    'ccp_alpha': [0.0, best_alpha],
}

num_param_combinations = np.product(list(map(len, hyperparams.values())))
total_mean_scores = np.zeros((num_param_combinations, 2))
best_classifiers = []
for _ in range(20):
    dt = DecisionTreeClassifier(ccp_alpha=best_alpha)
    clf = GridSearchCV(dt, param_grid=hyperparams, cv=10)
    clf.fit(X, y)
    cv_results_df = pd.DataFrame(clf.cv_results_)
    total_mean_scores += cv_results_df[['mean_test_score', 'std_test_score']]
    best_classifiers.append(clf.best_estimator_)

total_mean_scores /= 20
params = cv_results_df[[f'param_{p}' for p in hyperparams.keys()]]
results = pd.concat([params, total_mean_scores], axis=1)
print(results)

best_classifiers_average_test = sum(clf.score(test_X, test_y) for clf in best_classifiers) / 20
print(f'\nBest classifier average test accuracy: {best_classifiers_average_test}')

# TODO display best trees

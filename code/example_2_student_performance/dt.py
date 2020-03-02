from explore_dataset import load_preproccessed_dataset, get_raw_data

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz

(X, y), (test_X, test_y) = load_preproccessed_dataset(test_split=0.1, include_grades=True)

# Calculating the ccp_alpha that gives the best test score and prevents over-fitting
dt = DecisionTreeClassifier(random_state=42)
path = dt.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X, y)
    clfs.append(clf)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(ccp_alphas, [t.get_n_leaves() for t in clfs], drawstyle='steps-post')
ax1.set_xlabel("alpha")
ax1.set_ylabel("Number of nodes")
ax1.set_title("No. nodes vs alpha")

ax2.plot(ccp_alphas, [t.get_depth() for t in clfs], drawstyle='steps-post')
ax2.set_xlabel("alpha")
ax2.set_ylabel("Depth")
ax2.set_title("Depth of tree vs alpha")

train_scores = [clf.score(X, y) for clf in clfs]
test_scores = [clf.score(test_X, test_y) for clf in clfs]

best_alpha = ccp_alphas[np.argmax(test_scores)]

fig2, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()

hyperparams = {
    'max_depth': [3, 5, None],
    'ccp_alpha': [0.0, best_alpha],
}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid=hyperparams, cv=10)
clf.fit(X, y)

cv_results_df = pd.DataFrame(clf.cv_results_)
print(cv_results_df[[*(f'param_{p}' for p in hyperparams.keys()), 'mean_fit_time', 'mean_test_score', 'std_test_score']])

# Plot best decision tree with and without ccp
features = get_raw_data().keys()[:-1]
class_names = ['F', 'D', 'C', 'B', 'A']

dt_no_ccp = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_ccp = DecisionTreeClassifier(random_state=42, max_depth=3, ccp_alpha=best_alpha)
for dt in [dt_no_ccp, dt_ccp]:
    dt.fit(X, y)
    dot_data = tree.export_graphviz(
        dt,
        feature_names=features,
        class_names=class_names,
        special_characters=True,
        proportion=True,
        impurity=False,
        filled=True,
    )
    graphviz.Source(dot_data).view()

plt.show()

(math_x, math_y), _ = load_preproccessed_dataset(test_split=0.0, include_grades=True, subject='mat')

no_ccp_acc = sum(dt_no_ccp.predict(math_x) == math_y)/len(math_y)
ccp_acc = sum(dt_ccp.predict(math_x) == math_y)/len(math_y)

print(f'Maths accuracy: With pruning {ccp_acc*100:.2f}%, Without pruning {no_ccp_acc*100:.2f}%')

from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from explore_dataset import load_preproccessed_dataset
import matplotlib.pyplot as plt
import numpy as np

(X, y), _ = load_preproccessed_dataset(test_split=0.0, include_grades=True)

standardise = StandardScaler(copy=False)
standardise.fit_transform(X)

explained_variances = {}

pca = PCA()
pca.fit(X)
pca.explained_variance_ratio_.cumsum()
num_pc = len(pca.explained_variance_ratio_)

# Kaiser Stopping rule
print(pca.explained_variance_[:15])  # print top 15 eigenvalues

# Scree test
plt.figure(1)
plt.plot(pca.explained_variance_)
plt.xticks(list(range(1, len(pca.explained_variance_)+1, 2)))
plt.grid()
plt.xlabel('Principle Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Test')

fig, axis = plt.subplots(1, 1)
axis.plot(pca.explained_variance_ratio_.cumsum())
axis.plot(np.cumsum([1/num_pc for _ in range(num_pc)]), '--')
axis.set_xlabel('Principle components')
axis.set_ylabel('Variance')
axis.set_title("Accumulative explained variance")

plt.show()


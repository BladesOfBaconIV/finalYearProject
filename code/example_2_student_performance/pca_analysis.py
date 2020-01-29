from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from explore_dataset import load_preproccessed_dataset
import matplotlib.pyplot as plt
import numpy as np

(X, y), _ = load_preproccessed_dataset(test_split=0.0)

standardise = StandardScaler(copy=False)
standardise.fit_transform(X)

explained_variances = {}

pca = PCA()
pca.fit(X)
explained_variances['PCA'] = pca.explained_variance_ratio_.cumsum()

# “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
for k in ['poly', 'rbf', 'sigmoid']:
    kernel_pca = KernelPCA(kernel=k)
    X_ = kernel_pca.fit_transform(X)
    explained_variance = np.var(X_, axis=0)
    explained_variances[k] = np.cumsum(explained_variance / np.sum(explained_variance))

sparse_pca = SparsePCA()
X_ = sparse_pca.fit_transform(X)
explained_variance = np.var(X_, axis=0)
explained_variances['Sparse'] = np.cumsum(explained_variance / np.sum(explained_variance))

fig, axes = plt.subplots(1, 5)
for (key, variances), axis in zip(explained_variances.items(), axes):
    axis.plot(variances)
    axis.axhline(0.99)
    axis.set_xlabel('Dimensions')
    axis.set_ylabel('Variance')
    axis.set_title(key)

plt.show()
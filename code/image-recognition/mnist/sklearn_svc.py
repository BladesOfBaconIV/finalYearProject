from keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_train, img_rows, img_cols = X_train.shape
num_test, *_ = X_test.shape

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

svc = SVC()
params = {
    'kernel': ('linear', 'poly', 'rbf'),
    'gamma': ['scale', 0.1, 1, 10],
    'C': [1, 3, 5, 10],
}
gs = GridSearchCV(svc, params, cv=5, iid=True, verbose=2)
gs.fit(X_train[:2000], y_train[:2000])

best_svc = gs.best_estimator_
best_svc.fit(X_train, y_train)
print(best_svc.score(X_test, y_test))

with open('../../models/svc_gridsearch.pkl', 'wb') as f:
    pickle.dump(svc, f)   

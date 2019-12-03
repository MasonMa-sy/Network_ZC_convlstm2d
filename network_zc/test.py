import numpy as np
# a = [[[i+(k*4+j)*3 for i in range(3)] for j in range(4)] for k in range(5)]
# print(a)
# print(np.sum(a, axis=1))
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=(0, 1)))
from sklearn.model_selection import KFold

from network_zc.tools import data_preprocess, file_helper_unformatted, index_calculation

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

kf = KFold(n_splits=2)
for train_index, test_index in kf.split(X):
    print('train_index', train_index, 'test_index', test_index)
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]

    print(np.append(train_index, test_index))
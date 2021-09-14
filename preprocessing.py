import numpy as np
import math
import torch
from sklearn.preprocessing import StandardScaler
from random import sample

def load():
    ############# data preparation ############
    data = np.loadtxt('data.csv', delimiter=",", dtype=np.float32, skiprows=1)
    n_features = data.shape[1]-1
    n_samples = data.shape[0]

    #train and test sets
    train_idx = sample(range(n_samples), math.floor(n_samples*0.9))
    test_idx = [i for i in range(n_samples) if i not in train_idx]

    #division
    train_X = data[train_idx, 0:n_features]
    test_X = data[test_idx, 0:n_features]

    #scalling
    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    test_X = sc.transform(test_X)

    #conversion to tensors
    train_X = torch.from_numpy(train_X.astype(np.float32))
    test_X = torch.from_numpy(test_X.astype(np.float32))
    train_Y = torch.from_numpy(data[train_idx, -1].astype(np.float32))
    test_Y = torch.from_numpy(data[test_idx, -1].astype(np.float32))

    return train_X, train_Y, test_X, test_Y
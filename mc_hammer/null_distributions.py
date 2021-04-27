import numpy as np
from sklearn.decomposition import PCA
from statistics import stdev
import random

def random_order(arr, seed):
    np.random.seed(seed)
    new_arr = arr.copy()
    [np.random.shuffle(i) for i in new_arr.T]
    return(new_arr)

def min_max(arr,seed):
    np.random.seed(seed)
    new_arr = [[random.uniform(min(i),max(i)) for j in i] for i in arr.T]
    return(np.array(new_arr).T)


def pca_trans(arr,seed):
    np.random.seed(seed)
    pca = PCA()
    pca = pca.fit(arr)
    eig = pca.components_
    tm = pca.transform(arr)
    gaus_arr = np.empty([0,tm.shape[0]])
    for i in tm.T:
        sd = stdev(i)
        add = [random.gauss(0,sd) for j in i]
        gaus_arr = np.vstack([gaus_arr,add])
    return(np.dot(gaus_arr.T,eig))


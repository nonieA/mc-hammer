from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

def c_mat_maker(labels):
    c_mat = [(0 if i == j else 1 ) for i in labels for j in labels]
    c_mat = np.array(c_mat)
    shape = (len(labels),len(labels))
    return c_mat.reshape(shape)

def huberts_gamma(x,labels):
    p_mat = pairwise_distances(x)
    c_mat = c_mat_maker(labels)
    mat_sum = sum(p_mat[i,j] * c_mat[i,j] for i in range(len(p_mat)) for j in range(1,len(p_mat)) if j > i)
    m = (len(p_mat) * (len(p_mat) -1))/ 2
    return mat_sum/m

def norm_gamma(x,labels):
    p_mat = pairwise_distances(x)
    c_mat = c_mat_maker(labels)
    m = (len(p_mat) * (len(p_mat) - 1)) / 2

    p_mean = (sum(p_mat[i,j] for i in range(len(p_mat)) for j in range(1,len(p_mat)) if j > i))/m
    c_mean = (sum(c_mat[i,j] for i in range(len(c_mat)) for j in range(1,len(c_mat)) if j > i))/m
    p_var = ((sum(p_mat[i,j]**2 - p_mean**2
                 for i in range(len(p_mat)) for j in range(1,len(p_mat))
                 if j > i))/m)**(1/2)
    c_var = ((sum(p_mat[i,j]**2 - p_mean**2
                 for i in range(len(p_mat)) for j in range(1,len(p_mat))
                 if j > i))/m)**(1/2)
    mat_sum = sum((p_mat[i, j] - p_mean) * (c_mat[i, j] - c_mean)
                  for i in range(len(p_mat)) for j in range(1,len(p_mat))
                  if j > i)
    return (mat_sum/m)/(c_var * p_var)

def sillhouette(x,labels,distance):
    sil = silhouette_score(
        X=x,
        labels=labels,
        metric=distance
    )
    return sil

def CH(x,labels):
    ch_score = calinski_harabasz_score(x,labels)
    return ch_score

def DB(x,labels):
    db_score = davies_bouldin_score(x,labels)
    return db_score




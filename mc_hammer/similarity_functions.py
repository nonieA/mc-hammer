from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from s_dbw import S_Dbw, SD
from scipy.spatial import distance

def mean_cluster_diam(dist, labels,k):
    k_ind = [ind for ind,i in enumerate(labels) if i == k ]
    one_c = dist[k_ind]
    one_c = one_c[:,k_ind]
    diam = np.sum(one_c)/(len(k_ind)**2 - len(k_ind))
    return(diam)

def c_mat_maker(labels):
    c_mat = [(0 if i == j else 1 ) for i in labels for j in labels]
    c_mat = np.array(c_mat)
    shape = (len(labels),len(labels))
    return c_mat.reshape(shape)

def closest_neighbour(arr):
    arr_copy = arr.copy()
    arr_copy.sort()
    min_val = arr_copy[1]
    index = np.where(arr == min_val)[0][0]
    return index

def same_clust_ratio(labels,same_clust_list,k):
    nk = len(labels[labels == k])
    nn_count = [i for ind, i in enumerate(same_clust_list) if labels[ind] == k]
    return sum(nn_count)/nk

def bwc_k(x,labels,k,centers):
    labels_ind = [ind for ind,i in enumerate(labels) if i ==k]
    x_k = x[labels_ind,]
    mean_dist = np.mean(np.apply_along_axis(distance.euclidean, 1,x_k,centers[k]))
    cen_dist = pairwise_distances(centers)
    min_dist = min(cen_dist[np.nonzero(cen_dist)])
    return (min_dist-mean_dist)/max(min_dist,mean_dist)

def ind_sep_k(labels,arr,knn,c):
    k_nearest_n = sorted(range(len(arr)), key= lambda x:arr[x])[1:knn +1]
    ind_sep = sum([1 if labels[i] != c else 0 for i in k_nearest_n])/knn
    return ind_sep

def ind_sep_clust(labels,dist,knn,c):
    labels_ind = [ind for ind, i in enumerate(labels) if i ==c]
    sep_nc = sum([ind_sep_k(labels,dist[i],knn,c) for i in labels_ind])/len(labels_ind)
    return sep_nc

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

def sillhouette_euclidean(x,labels):
    sil = silhouette_score(
        X=x,
        labels=labels,
        metric='euclidean'
    )
    return sil


def sillhouette_cosine(x,labels):
    sil = silhouette_score(
        X=x,
        labels=labels,
        metric='cosine'
    )
    return sil


def CH(x,labels):
    ch_score = calinski_harabasz_score(x,labels)
    return ch_score

def DB(x,labels):
    db_score = davies_bouldin_score(x,labels)
    return db_score

def dunn(x,labels,centers):
    dist = pairwise_distances(x)
    cen_dist = pairwise_distances(centers)
    min_dist = min(cen_dist[np.nonzero(cen_dist)])
    max_diam = max([mean_cluster_diam(dist,labels,i) for i in np.unique(labels)])
    return min_dist/max_diam

def S_DBW(x,labels):
    return S_Dbw(x, labels, centers_id=None, method='Tong', alg_noise='bind',
                 centr='mean', nearest_centr=True, metric='euclidean')

def SD_score(x,labels):
    return SD(x, labels, k=1.0, centers_id=None,  alg_noise='bind',centr='mean', nearest_centr=True, metric='euclidean')

def IGP(x,labels):
    dist = pairwise_distances(x)
    nearest_n = np.array([range(len(labels)),[closest_neighbour(dist[i]) for i in range(len(labels))]])
    same_clust_list = [1 if labels[nearest_n[0,i]] == labels[nearest_n[1,i]] else 0 for i in range(len(labels))]
    igp_c = [same_clust_ratio(labels,same_clust_list,i) for i in np.unique(labels)]
    return(np.mean(igp_c))

def BWC(x,labels,centers):
    bwc_c = mean([bwc_k(x,labels,i,centers) for i in np.unique(labels)])
    return(bwc_c)

def CVNN(x,labels):
    nc = len(np.unique(labels))
    knn_range = range(3,round(len(labels)/(nc*3)))
    dist = pairwise_distances(x)
    sep_range = [[ind_sep_clust(labels,dist,j,i) for i in range(nc)] for j in knn_range]
    sep_range = [np.mean(i)/max(i) for i in sep_range]
    sep = min(sep_range)
    comp_list = [mean_cluster_diam(dist,labels,i) * 2 for i in range(nc)]
    comp = np.mean(comp_list)/max(comp_list)
    return sep + comp

if __name__ == '__main__':
    x = np.random.rand(100,3)
    km = KMeans(n_clusters = 3).fit(x)
    labels = km.labels_
    centers = km.cluster_centers_

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from mc_hammer.similarity_functions import ind_sep_clust
from mc_hammer.clustering_algorithms import k_means

def clust_dist(x,labels,k):
    one_clust = x[labels ==k,]
    return pairwise_distances(one_clust)

def mean_clust_dist(clust_d):
    mean_d = np.sum(clust_d) / (len(k_ind) ** 2 - len(k_ind))
    return mean_d

def centre_dist(x,labels,k,centers):
    one_clust = x[labels == k,]
    dist_list = np.apply_along_axis(distance.euclidean, 1, one_clust, centers[k])
    return dist_list

def mid_points(centers,k):
    center_copy = centers.copy()
    centers_drop = np.delete(center_copy,k,0)
    mid_points = np.array([(centers[k] + i)/2 for i in centers_drop])
    return mid_points

def clust_sd(x,labels,k):
    one_clust = x[labels == k,]
    sd = np.std(one_clust,0)
    sd_T = np.dot(sd.T,sd)**0.5
    return sd_T

def full_sd(x,labels):
    sd = (sum([clust_sd(x,labels,i) for  i in range(max(labels) +1)]))**0.5
    return sd/max(labels)

def mean_center_dist(x,labels,centers):
    """
    measure of compactness through mean distance for each point, to the center
    :param x:
    :param labels:
    :param centers:
    :return:
    """
    mean_list = [np.mean(centre_dist(x,labels,i,centers)) for i in range(len(centers))]
    return mean_list

def max_center_dist(x,labels,centers):
    """
    measure pf compactness, max distance between each point and cthe cluster center
    :param x:
    :param labels:
    :param centers:
    :return:
    """
    max_list = [max(centre_dist(x,labels,i,centers)) for i in range(len(centers))]
    return max_list

def max_diam(x,labels,centers):
    """
    measure of compactness, max distance between points in a cluster
    :param x:
    :param labels:
    :return:
    """
    dist_list = [clust_dist(x,labels,i) for i in range(len(centers))]
    max_list = [i.max() for i in dist_list]
    return max_list

def mean_max_diam(x,labels,centers):
    """
    measure of compactness, mean of the max distance of each point to each other point
    :param x:
    :param labels:
    :return:
    """
    dist_list = [clust_dist(x, labels, i) for i in range(len(centers))]
    mean_list = [np.mean(np.apply_along_axis(max,1,i)) for i in dist_list]
    return mean_list

def mean_all(x,labels,centers):
    """
    measure of compactness mean distance of all points to all points
    :param x:
    :param labels:
    :return:
    """
    dist_list = [clust_dist(x, labels, i) for i in range(len(centers))]
    mean_list = [i.mean() for i in dist_list]
    return mean_list

def radial_density1p(x,sd,ref_point):
    dist_point = [distance.euclidean(i,ref_point) for i in x]
    dens = len([i for i in dist_point if i <= sd])
    return dens

def radial_density(x,centers,labels,measure):
    """
    measure of compactness densitt from center ratio density inbetween clusters
    :param x:
    :param centers:
    :param labels:
    :param measure:
    :return:
    """
    k = max(labels)
    cn = k-1
    sd = full_sd(x,labels)
    mid_point_list = [mid_points(centers,i) for i in range(max(labels) +1)]
    mid_point_list = [j for i in mid_point_list for j in i]
    centers_dense = [radial_density1p(x,sd,i)for i in centers]
    mid_point_dense = [radial_density1p(x,sd,i) for i in mid_point_list]
    if measure == 'single_clusters_max':
        dens_list = []
        if k == 1:
            return mid_point_dense[0]
        else:
            for i in range(k + 1):
                mid_dense = max(mid_point_dense[i * k:(i + 1) * k])
                dens_list.append((mid_dense / centers_dense[i]))
            return dens_list
    elif measure == 'single_cluster_mean':
        if k == 1:
            return mid_point_dense[0]
        else:
            dens_list = []
            for i in range(k + 1):
                mid_dense = np.mean(mid_point_dense[i * k:(i + 1) * k])
                dens_list.append((mid_dense / centers_dense[i]))
            return dens_list
    else:
        if k == 1:
            return mid_point_dense[0] / ((k+1) * k)
        else:
            dens_list = []
            for i in range(max(labels) + 1):
                mid_dense = max(mid_point_dense[i * k:(i + 1) * k])
                dens_list.append((mid_dense / centers_dense[i]))
            return sum(dens_list) / ((k+1) * k)

def clust_center_dist(centers):

    return pairwise_distances(centers)

def dataset_meancenter_dist(x,centers):
    """
    cluster seperation mean distance between cluster centers
    :param x:
    :param centers:
    :return:
    """
    d_center = x.mean(axis = 0)
    center_dist = [distance.euclidean(i,d_center) for i in centers]
    return center_dist

def dataset_midpoint_dist(x,centers):
    """
    seperation, distance from cluster centers to dataset midpoint
    :param x:
    :param centers:
    :return:
    """
    d_center = np.array([(max(x[:,i]) + min(x[:,i]))/2 for i in range(x.shape[1])])
    center_dist = [distance.euclidean(i, d_center) for i in centers]
    return center_dist

def scatter(x,labels):
    """
    seperation sd of cluster compared to sd of full dataset
    :param x:
    :param labels:
    :return:
    """
    sd = np.std(x,0)
    sd_D = np.dot(sd.T,sd)**0.5
    sd_C = sum([clust_sd(x,labels,i) for i in range(max(labels)+1)])
    scat = (sd_C/sd_D)/max(labels)
    return scat

def cvnn_sep(x,labels):
    """
    seperation distances of based on points on cluster edges
    :param x:
    :param labels:
    :return:
    """
    nc = len(np.unique(labels))
    knn_range = range(3,round(len(labels)/(nc*3)))
    dist = pairwise_distances(x)
    sep_range = [[ind_sep_clust(labels, dist, j, i) for i in range(nc)] for j in knn_range]
    return sep_range

if __name__=='__main__':
    x = np.random.rand(100,3)
    km = KMeans(n_clusters = 4).fit(x)
    labels = km.labels_
    centers = km.cluster_centers_
    uni_dis = [np.random.rand(300,3) for j in range(100)]
    uni_dis = {'x':uni_dis,
               'labs':[k_means(j,2) for j in uni_dis]}
    uni_dis = {
        'x':uni_dis['x'],
        'labs':[i[0] for i in uni_dis['labs']],
        'centers':[j[1] for j in uni_dis['labs']]
    }


    def get_metrics_results(res_dicts, method, addit=None):
        res_list = []
        for i in range(len(res_dicts['x'])):
            x = res_dicts['x'][i]
            labs = res_dicts['labs'][i]
            centers = res_dicts['centers'][i]
            if method in ['mean_all', 'mean_max_diam','max_diam','mean_center_dist', 'max_center_dist']:
                res = eval(method + '(x,labs,centers)')
            elif method in ['scatter', 'cvnn_sep', 'IGP',
                            'sillhouette_euclidean']:
                res = eval(method + '(x,labs)')
            elif method in ['dataset_midpoint_dist', 'dataset_meancenter_dist']:
                res = eval(method + '(x,centers)')
            else:
                res = eval(method + '(x,centers,labs,addit)')
            res_list.append(res)
        return (res_list)

    method_list = ['cvnn_sep', 'scatter' ,'dataset_midpoint_dist', 'dataset_meancenter_dist','mean_center_dist','max_center_dist','max_diam','mean_max_diam','mean_all']
    for i in method_list:
        get_metrics_results(uni_dis,i)

    for i in ['single_cluster_max','single_cluster_mean','ratio']:
        get_metrics_results(uni_dis,'radial_density',addit=i)

    k = max(labels)
    cn = k - 1
    sd = full_sd(x, labels)
    mid_point_list = [mid_points(centers, i) for i in range(max(labels) + 1)]
    mid_point_list = [j for i in mid_point_list for j in i]
    centers_dense = [radial_density1p(x, sd, i) for i in centers]
    mid_point_dense = [radial_density1p(x, sd, i) for i in mid_point_list]
    if measure == 'single_clusters_max':
        dens_list = []
        if cn == 2:
            return mid_point_dense[0]
        else:
            for i in range(k + 1):
                mid_dense = max(mid_point_dense[i * k:(i + 1) * k])
                dens_list.append((mid_dense / centers_dense[i]))
            return dens_list
    elif measure == 'single_cluster_mean':
        if cn == 2:
            return mid_point_dense[0]
        else:
            dens_list = []
            for i in range(k + 1):
                mid_dense = np.mean(mid_point_dense[i * k:(i + 1) * k])
                dens_list.append((mid_dense / centers_dense[i]))
            return dens_list
    else:
        if cn == 2:
            return mid_point_dense[0] / ((k+1) * k)
        else:
            dens_list = []
            for i in range(max(labels) + 1):
                mid_dense = max(mid_point_dense[mid_point_dense[i * k:(i + 1) * k])
                dens_list.append((mid_dense / centers_dense[i]))
            return sum(dens_list) / ((k+1) * k)
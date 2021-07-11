import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from mc_hammer.similarity_functions import ind_sep_clust

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
    mean_list = [np.mean(centre_dist(x,labels,i,centers)) for i in range(len(centers))]
    return mean_list

def max_center_dict(x,labels,centers):
    max_list = [max(centre_dist(x,labels,i,centers)) for i in range(len(centers))]
    return max_list

def max_diam(x,labels):
    dist_list = [clust_dist(x,labels,i) for i in range(len(centers))]
    max_list = [i.max() for i in dist_list]
    return max_list

def mean_max_diam(x,labels):
    dist_list = [clust_dist(x, labels, i) for i in range(len(centers))]
    mean_list = [np.mean(np.apply_along_axis(max,1,i)) for i in dist_list]
    return mean_list

def mean_all(x,labels):
    dist_list = [clust_dist(x, labels, i) for i in range(len(centers))]
    mean_list = [mean_center_dist(i) for i in dist_list]
    return mean_list

def radial_density1p(x,sd,ref_point):
    dist_point = [distance.euclidean(i,ref_point) for i in x]
    dens = len([i for i in dist_point if i <= sd])
    return dens

def radial_desnsity(x,centers,labels,measure):
    k = max(labels)
    cn = k-1
    sd = full_sd(x,labels)
    mid_point_list = [mid_points(centers,i) for i in range(max(labels) +1)]
    mid_point_list = [j for i in mid_point_list for j in i]
    centers_dense = [radial_density1p(x,sd,i)for i in centers]
    mid_point_dense = [radial_density1p(x,sd,i) for i in mid_point_list]
    if measure == 'single_clusters_max':
        dens_list = []
        for i in range(k +1):
            mid_dense = max(mid_point_dense[i*cn:(i+1)*cn])
            dens_list.append((mid_dense/centers_dense[i]))
        return dens_list
    elif measure == 'single_cluster_mean':
        dens_list = []
        for i in range(k + 1):
            mid_dense = np.mean(mid_point_dense[i * cn:(i + 1) * cn])
            dens_list.append((mid_dense / centers_dense[i]))
        return dens_list
    else:
        dens_list = []
        for i in range(max(labels) +1):
            mid_dense = max(mid_point_dense[i*cn:(i+1)*cn])
            dens_list.append((mid_dense/centers_dense[i]))
        return sum(dens_list)/(k*cn)

def clust_center_dist(centers):
    return pairwise_distances(centers)

def dataset_meancenter_dist(x,centers):
    d_center = x.mean(axis = 0)
    center_dist = [distance.euclidean(i,d_center) for i in centers]
    return center_dist

def dataset_midpoint_dist(x,centers):
    d_center = np.array([(max(x[:,i]) + min(x[:,i]))/2 for i in range(x.shape[1])])
    center_dist = [distance.euclidean(i, d_center) for i in centers]
    return center_dist

def scatter(x,labels):
    sd = np.std(one_clust,0)
    sd_D = np.dot(sd.T,sd)**0.5
    sd_C = sum([clust_sd(x,labels,i) for i in range(max(labels)+1)])
    scat = (sd_C/sd_D)/max(labels)
    return scat

def cvnn_sep(x,labels):
    nc = len(np.unique(labels))
    knn_range = range(3,round(len(labels)/(nc*3)))
    dist = pairwise_distances(x)
    sep_range = [[ind_sep_clust(labels, dist, j, i) for i in range(nc)] for j in knn_range]
    return sep_range

if __name__=='__main__':
    x = np.random.rand(100,3)
    km = KMeans(n_clusters = 3).fit(x)
    labels = km.labels_
    centers = km.cluster_centers_
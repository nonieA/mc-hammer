from mc_hammer.null_distributions import pca_trans, random_order, min_max
from mc_hammer.clustering_algorithms import k_means,dbscan,spectral_clustering
from mc_hammer.similarity_functions import huberts_gamma, norm_gamma, sillhouette_euclidean,sillhouette_cosine, CH, DB
from mc_hammer.hypothesis_test import hypothesis_test
import numpy as np

class mchammer():
    def __init__(self):
        pass
    def get_null_distributions(self,x,null_method,repeats):
        if null_method == 'pca_trans':
            self._null_dists = [pca_trans(x,i) for i in range(repeats)]
        elif null_method == 'random_order':
            self._null_dists = [random_order(x, i) for i in range(repeats)]
        elif null_method == 'min_max':
            self._null_dists = [min_max(x, i) for i in range(repeats)]
        self._null_dists.append(x)
    def get_q_scores(self,cluster_method,q_methods = 'all',k = None, eps = None,min_samples = None):
        if q_methods == 'all':
            q_methods = ['huberts_gamma', 'norm_gamma', 'sillhouette_euclidean','sillhouette_cosine', 'CH', 'DB']
        if cluster_method == 'K_Means':
            labels = [k_means(i,k) for i in self._null_dists]
        if cluster_method == 'DBSCAN':
            labels = [dbscan(i, eps,min_samples) for i in self._null_dists]
        if cluster_method == 'spectral_clustering':
            labels = [spectral_clustering(i, k) for i in self._null_dists]

        q_dict = {}
        if 'huberts_gamma' in q_methods:
            q_dict['huberts_gamma'] = ['one_cluster' if len(set(labels[i])) == 1 else huberts_gamma(
                self._null_dists[i],
                labels[i]
            ) for i in range(len(labels))]
        if 'norm_gamma' in q_methods:
            q_dict['norm_gamma'] = ['one_cluster' if len(set(labels[i])) == 1 else norm_gamma(
                self._null_dists[i],
                labels[i]
            ) for i in range(len(labels))]
        if 'sillhouette_euclidean' in q_methods:
            q_dict['sillhouette_euclidean'] = ['one_cluster' if len(set(labels[i])) == 1 else sillhouette_euclidean(
                self._null_dists[i],
                labels[i]
            ) for i in range(len(labels))]
        if 'sillhouette_cosine' in q_methods:
            q_dict['sillhouette_cosine'] = ['one_cluster' if len(set(labels[i])) == 1 else sillhouette_cosine(
                self._null_dists[i],
                labels[i]
            ) for i in range(len(labels))]
        if 'CH' in q_methods:
            q_dict['CH'] = ['one_cluster' if len(set(labels[i])) == 1 else CH(
                self._null_dists[i],
                labels[i]
            ) for i in range(len(labels))]
        if 'DB' in q_methods:
            q_dict['DB'] = ['one_cluster' if len(set(labels[i])) == 1 else DB(
                self._null_dists[i],
                labels[i]
            ) for i in range(len(labels))]
        results_dict = {k:hypothesis_test(v,k) for k,v in q_dict.items()}
        return results_dict

if __name__ == '__main__':
    test = np.random.rand(100,3)

    def get_null_distributions(x,null_method,repeats):
        null_dists = [eval(null_method + '(x,' + str(i) + ')') for i in range(repeats)]
        null_dists.append(x)
        return null_dists

    mch = mchammer()
    mch.get_null_distributions(test,'min_max',3)
    mch.get_q_scores(cluster_method='K_Means',k =2)
    i = huberts_gamma
    null_dists = [np.random.rand(100,3),np.random.rand(100,3)]
    labels = [np.random.rand(100,1),np.random.rand(100,1)]
    eval('huberts_gammma(x_data,labals_data)',{"__builtins__":None},
         {
             x_data: null_dists[j],
             labels_data: labels[j]
         }
         )
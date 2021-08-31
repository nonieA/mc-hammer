from mc_hammer.null_distributions import pca_trans, random_order, min_max
from mc_hammer.clustering_algorithms import k_means,dbscan,spectral_clustering
from mc_hammer.similarity_functions import huberts_gamma, norm_gamma, sillhouette_euclidean,sillhouette_cosine, CH, DB
from mc_hammer.similarity_functions import dunn,S_Dbw,SD_score,IGP,BWC,CVNN, dunn_min
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
            q_methods = ['huberts_gamma', 'norm_gamma', 'sillhouette_euclidean','sillhouette_cosine', 'CH', 'DB',
                         'dunn','S_Dbw','SD_score','IGP','BWC','CVNN','dunn_min']
        if isinstance(q_methods,str):
            q_methods = [q_methods]
        if cluster_method == 'K_Means':
            labels = [k_means(i,k) for i in self._null_dists]
        if cluster_method == 'DBSCAN':
            labels = [dbscan(i, eps,min_samples) for i in self._null_dists]
        if cluster_method == 'spectral_clustering':
            labels = [spectral_clustering(i, k) for i in self._null_dists]
        self.x_labels = labels[-1]
 #       results_dict = {k:hypothesis_test(v,k) for k,v in q_dict.items()}
        q_dict = {}
        for i in q_methods:
            res = []
            if i in ['BWC','dunn','dunn_min']:
                for j in range(len(labels)):
                    res_small = eval(i + '(self._null_dists[' + str(j) + '],labels[' + str(j) + '][0],labels[' + str(j) + '][1])')
                    res.append(res_small)
            else:
                for j in range(len(labels)):
                    res_small = eval(i + '(self._null_dists['+str(j)+'],labels['+str(j)+'][0])')
                    res.append(res_small)
            q_dict[i] = res
        results_dict = {k: hypothesis_test(v, k) for k, v in q_dict.items()}

        return results_dict

if __name__ == '__main__':
    test = np.random.rand(100,3)


    mch = mchammer()
    mch.get_null_distributions(test,'min_max',3)
    res_q = mch.get_q_scores(cluster_method='K_Means',k =2)
    seperation = 0.1
    noise = 0.1
    k_test = [make_classification(
        n_samples = 1000,
        n_features = 10,
        n_informative = int((10-10*noise)),
        n_redundant = int(10*noise),
        n_classes = 4,
        n_clusters_per_class = 1,
        class_sep = seperation,
        random_state = i
    )[0] for i in range(100)]

    for ind,i in enumerate(k_test):
        print(ind)
        mch = mchammer()
        mch.get_null_distributions(k_test[84],'pca_trans',100)
        res_q = mch.get_q_scores(cluster_method='K_Means',k =4)
        print(res_q)
    x = k_test[84]
    repeats = 100
    null_dists = [pca_trans(x, i) for i in range(repeats)]
    labels = [k_means(i,4) for i in null_dists]

    q_methods = ['huberts_gamma', 'norm_gamma', 'sillhouette_euclidean', 'sillhouette_cosine', 'CH', 'DB',
                 'dunn', 'S_Dbw', 'SD_score', 'IGP', 'BWC', 'CVNN']

    q_dict = {}
    for i in q_methods:
        print(i)
        res = []
        if i in ['BWC', 'dunn']:
            for j in range(len(labels)):
                print(j)
                res_small = eval(
                    i + '(null_dists[' + str(j) + '],labels[' + str(j) + '][0],labels[' + str(j) + '][1])')
                res.append(res_small)
        else:
            for j in range(len(labels)):
                print(j)
                res_small = eval(i + '(null_dists[' + str(j) + '],labels[' + str(j) + '][0])')
                res.append(res_small)
        q_dict[i] = res
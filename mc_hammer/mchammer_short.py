import numpy as np
from sklearn.metrics import calinski_harabasz_score
import random
import numpy as np
import math
from sklearn.cluster import  KMeans


def k_means(x,k):
    kmeans = KMeans(n_clusters=k,random_state=4).fit(x)
    return kmeans.labels_, kmeans.cluster_centers_

def hypothesis_test(q_list,q_method):
    q_no_miss = [i for i in q_list if isinstance(i,str) == False]
    if (len(q_no_miss) == 0) or (math.isnan(q_list[-1])):
        return 0.99
    else:
        q_list = [max(q_no_miss) if i == 'one_cluster' else i for i in q_list]
        x_val = q_list[-1]
        q_arr = np.sort(np.array(q_list))
        p_val = (np.where(q_arr == x_val)[0][0] + 1)/len(q_list)
        if q_method in ['DB','CVNN','SD_score','S_Dbw']:
            if q_list[-1] == max(q_list):
                return 0.99
            else:
                return p_val

        else:
            if q_list[-1] == min(q_list):
                return 0.99
            else:
                return 1-p_val

def CH(x,labels):
    ch_score = calinski_harabasz_score(x,labels)
    return ch_score

def min_max(arr,seed):
    np.random.seed(seed)
    new_arr = [[random.uniform(min(i),max(i)) for j in i] for i in arr.T]
    return(np.array(new_arr).T)


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
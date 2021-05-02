from sklearn.cluster import DBSCAN, KMeans, SpectralClustering


def k_means(x,k):
    kmeans = KMeans(n_clusters=k,random_state=4).fit(x)
    return kmeans.labels_

def dbscan(x,eps,min_samples):
    db = DBSCAN(eps = eps,min_samples=min_samples).fit(x)
    return db.labels_

def spectral_clustering(x,k):
    sc = SpectralClustering(n_clusters=k).fit(x)
    return sc.labels_

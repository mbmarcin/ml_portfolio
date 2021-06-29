import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from kneed import KneeLocator


class Ml_tools:
    """
    XXX
    """

    def __init__(self, df_X, df_Y):
        self.df_x = df_X
        self.df_y = df_Y

    def scaler(self, mode_scaler):
        """
        XXX
        """
        if mode_scaler == 0:
            return MinMaxScaler().fit_transform(self.df_x)
        elif mode_scaler == 1:
            return StandardScaler().fit_transform(self.df_x)
        else:
            print(f'Wrong parameter')

    def find_n_comps_usePCA(self, df_, exp_var_=0.8):
        """
        XXX
        """
        print(f'info -> Use scale or standardize or normalize')
        pca = PCA()
        pca.fit(df_)
        evr = pca.explained_variance_ratio_
        for i, exp_var in enumerate(evr.cumsum()):
            if exp_var >= exp_var_:
                n_comps = i + 1
                break
        print(f"Finding optimal number of components {n_comps}, explained_variance_max {exp_var_}")
        pca = PCA(n_components=n_comps)
        pca.fit(df_)
        scores_pca = pca.transform(df_)
        return scores_pca

    def clustering(self, dta, algorithm_='auto', copy_x_=True, init_='k-means++', max_iter_=300,
                   n_clusters_=3, n_init_=10, n_jobs_=20, precompute_distances_='auto',
                   random_state_=2020, tol_=0.0001, verbose_=0, max_clusters_=12):
        sse = list()
        silhouette_coef_kmeans = list()
        silhouette_coef_agglo = list()
        max_clusters = max_clusters_
        for i in range(1, max_clusters):
            # KMEANS
            km = KMeans(algorithm=algorithm_, copy_x=copy_x_, init=init_, max_iter=max_iter_,
                        n_clusters=i, n_init=n_init_, n_jobs=n_jobs_, precompute_distances=precompute_distances_,
                        random_state=random_state_, tol=tol_, verbose=verbose_)
            km.fit(dta)
            sse.append(km.inertia_)

            # score = silhouette_score(dta, km.labels_, metric='euclidean')
            # silhouette_coef_kmeans.append(score)
            #
            # # AgglomerativeClustering
            # cluster_model = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
            # cluster_labels = cluster_model.fit_predict(dta)
            # silhouette_avg = silhouette_score(dta, cluster_labels, metric='euclidean')
            # silhouette_coef_agglo.append(silhouette_avg)

        # n_clusters = KneeLocator([i for i in range(2, max_clusters)], sse, curve='convex', direction='decreasing').elbow
        n_clusters = KneeLocator(range(1, max_clusters), sse, curve='convex', direction='decreasing').elbow
        print(f'Optimal k-value: {n_clusters}')

        km = KMeans(algorithm=algorithm_, copy_x=copy_x_, init=init_, max_iter=max_iter_,
                    n_clusters=n_clusters, n_init=n_init_, n_jobs=n_jobs_, precompute_distances=precompute_distances_,
                    random_state=random_state_, tol=tol_, verbose=verbose_)
        km.fit(dta)

        # distance
        # dists = euclidean_distances(km.cluster_centers_)

        mm = {'sse': sse,
              'n_clusters': n_clusters,
              'labels': km.labels_}

        return mm #silhouette_coef_kmeans, silhouette_coef_agglo

# test = Ml_tools(dfX, dfY)
#
# df = test.scaler(1)
#
# metrics = test.clustering(df)
#
# x0 = metrics[0]
# x1 = metrics[1]
#
# import matplotlib.pyplot as plt
#
# plt.plot(x0)
# plt.ylabel('some numbers')
# plt.title('KMEANS')
# plt.show()
# print(sorted(x0))
# print(x0)
#
# plt.plot(x1)
# plt.title('AgglomerativeClustering')
# plt.ylabel('some numbers')
# plt.show()

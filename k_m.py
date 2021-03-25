import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px


# %matplotlib inline
# df = pd.read_excel('tabela_DH5.xlsx', index_col=[0], header=[0,1])
# df = pd.read_excel('tabela_DH5.xlsx', header=[0,1])
# df.loc[89, :]

class Clusters:

    def __init__(self, dta, algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                 n_clusters=3, n_init=10, n_jobs=20, precompute_distances='auto',
                 random_state=2020, tol=0.0001, verbose=0):
        self.algorithm = algorithm
        self.copy_x = copy_x
        self.init = init
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.precompute_distances = precompute_distances
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.df = dta
        self.X = MinMaxScaler().fit_transform(self.df.drop(self.df.columns[0], axis=1).fillna(0))

    # def prep_df():
    #     scale = MinMaxScaler()
    #     print('ok')
    #     return scale.fit_transform(self.df.drop(self.df.columns[0], axis=1).fillna(0))
    # X = self.prep_df()

    def core_algo(self):
        km = KMeans(algorithm=self.algorithm, copy_x=self.copy_x, init=self.init, max_iter=self.max_iter,
                    n_clusters=self.n_clusters, n_init=self.n_init, n_jobs=self.n_jobs,
                    precompute_distances=self.precompute_distances,
                    random_state=self.random_state, tol=self.tol, verbose=self.verbose)
        return km

    def get_k_value(self):
        silhouette_coefficients = []
        inertia = []

        # func to find distance between a point and line in 2-d
        def calc_distance(x1, y1, a, b, c):
            d = abs((a * x1 + b * y1 + c)) / (np.sqrt(a * a + b * b))
            return d

        K = range(2, 15)

        for k in K:
            algo = KMeans(algorithm=self.algorithm, copy_x=self.copy_x, init=self.init, max_iter=self.max_iter,
                          n_clusters=k, n_init=self.n_init, n_jobs=self.n_jobs,
                          precompute_distances=self.precompute_distances,
                          random_state=self.random_state, tol=self.tol, verbose=self.verbose)
            algo.fit(self.X)
            score = silhouette_score(self.X, algo.labels_)
            silhouette_coefficients.append(score)
            inertia.append(algo.inertia_)

        a = inertia[0] - inertia[12]
        b = K[12] - K[0]
        c1 = K[0] * inertia[12]
        c2 = K[12] * inertia[0]
        c = c1 - c2

        distance_of_points_from_line = []
        for k in range(13):
            distance_of_points_from_line.append(calc_distance(K[k], inertia[k], a, b, c))

        f = plt.figure(figsize=(20, 10))

        ax = f.add_subplot(221)
        ax.plot(K, silhouette_coefficients)
        ax.title.set_text('The S Method')  # Silhouette
        ax.set_xlabel('No of clusters')

        ax2 = f.add_subplot(222)
        ax2.plot(K, inertia)
        ax2.title.set_text('The E Method')  # Elbow
        ax2.set_xlabel('No of clusters')
        # plt.show()

        ax3 = f.add_subplot(224)
        ax3.plot(K, distance_of_points_from_line)
        # ax3.title.set_text('X')
        ax3.set_xlabel('No of clusters')
        plt.savefig('method.png')
        plt.show()

    def get_radar_chart_clusters(self):
        algos = self.core_algo()
        algos.fit(self.X)
        clusters = pd.DataFrame(self.X, columns=self.df.columns[1:])
        clusters['label'] = algos.labels_
        # clusters[self.df.columns[0]] = self.df.iloc[:, 0]
        polar = clusters.groupby("label").mean().reset_index()
        polar = pd.melt(polar, id_vars=["label"])
        polar.loc[:, :].to_csv('df_polar.txt', sep=';', index=False, header=True)
        fig = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True, height=700, width=1500)
        fig.show()
        self.df['c'] = algos.labels_
        self.df.loc[:, :].to_csv('df_oe.txt', sep=';', index=False, header=True)

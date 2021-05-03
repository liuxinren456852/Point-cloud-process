import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import KMeans
from sklearn import neighbors

plt.style.use('seaborn')


class spectral_clustering(object):
    def __init__(self, n_clusters, epsilon=1e-4, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.epsilon = epsilon


        self.W = None           # 图的相似性矩阵
        self.L = None           # 图的拉普拉斯矩阵
        self.D = None           # 图的度矩阵

        self.N = None
        self.vector = None

    def init_param(self, data):
        # 初始化参数
        self.N = data.shape[0]      
        self.cal_weight_mat(data)               # 计算相似矩阵W
        self.D = np.diag(self.W.sum(axis=1))    # 计算度矩阵D
        self.L = self.D - self.W                # 计算未归一化的拉普拉斯矩阵
        return


    # 计算相似矩阵W
    def cal_weight_mat(self, data, n_neighbors=5):
        self.W = neighbors.kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
        self.W = np.array(self.W.A)
        self.W = 0.5 * (self.W + self.W.T)

    def fit(self, data):

        self.init_param(data)
        w, v = np.linalg.eig(self.L)                # 特征值分解
        inds = np.argsort(w)[:self.n_clusters]      # 选取前k个特征向量
        self.vector = v[:, inds]                    # 构造新的k维数据

    def predict(self, data):
        
        # 对新的k维数据进行k均值聚类
        km = KMeans.K_Means(n_clusters=self.n_clusters, tolerance=self.epsilon, max_iter=self.max_iter)
        km.fit(self.vector)
        result = km.predict(self.vector)

        return result


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    s_clustering = spectral_clustering(n_clusters=3)
    s_clustering.fit(X)
    cat = s_clustering.predict(X)
    print(cat)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5, c=cat)
    plt.show()
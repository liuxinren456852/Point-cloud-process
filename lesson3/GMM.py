# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.Mu = None          #k个类的均值
        self.Var = None         #k个类的协方差
        self.Pi = None          #k个类的权重值
        self.W = None           #r(Z_nk)
        self.data = None        #数据点
        self.n_points = None    #数据点个数
        self.loglh = None       #损失函数
    
    #利用数据初始化相关参数
    def initialize(self, data):

        self.n_points = data.shape[0]
        self.data = data
        self.Mu = np.empty((0,data.shape[1]))
        self.Var = []


        # 随机选取k个点
        for index in np.random.choice(self.n_points,self.n_clusters):

            self.Mu = np.append(self.Mu,[data[index,:]],axis=0)


            self.Var.append(10 * np.diag([1,1]))
        
        self.W = np.ones((self.n_points,self.n_clusters))/self.n_clusters
        self.Pi = [1/self.n_clusters] * self.n_clusters

        self.loglh = []

    # 更新r(z_nk)
    def update_W(self):

        pdfs = np.zeros(((self.n_points, self.n_clusters)))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf( self.data, self.Mu[i], np.asarray(self.Var[i]) )
        self.W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return self.W    

    # 更新Mu
    def update_Mu(self):
        self.Mu = np.zeros((self.n_clusters, self.data.shape[1]))
        for i in range(self.n_clusters):
            self.Mu[i] = np.average(self.data, axis=0, weights=self.W[:, i])

    # 更新Var
    def update_Var(self):
        self.Var = []
        for i in range(self.n_clusters):
            self.Var.append(np.cov(self.data - self.Mu[i], rowvar=0, aweights=self.W[:, i]))


    # 更新pi
    def update_Pi(self):
        self.Pi = self.W.sum(axis=0) / self.n_points
        return self.Pi
        

    # 计算损失函数
    def logLH(self):
        # pi_k * N(x_n|mu,sigma)
        pdfs = np.zeros(((self.n_points, self.n_clusters)))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(self.data, self.Mu[i], self.Var[i])
        return np.sum(np.log(pdfs.sum(axis=1)),axis=0)
    
    def fit(self, data):
        
        # 初始化相关参数
        self.initialize(data)

        # 迭代次数
        num_iter = 0
        
        # 保存当前的损失函数
        self.loglh.append(self.logLH())


        while num_iter < self.max_iter:
            self.update_W()
            self.update_Pi()
            self.update_Mu()
            self.update_Var()

            self.loglh.append(self.logLH())
            # print(self.loglh)
            # if abs(self.loglh[-1] - self.loglh[-2]) < 1e-9:
            #     break
            num_iter += 1

    
    def predict(self, data):
        # 屏蔽开始
        result = []

        pdfs = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(data, self.Mu[i], np.diag(self.Var[i]))
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        # print(W)
        result = np.argmax(W, axis=1)
        # 屏蔽结束
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

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)

    # print(cat)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5, c=cat)
    plt.show()
    # 初始化

    


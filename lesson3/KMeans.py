# 文件功能： 实现 K-Means 算法

import numpy as np
import math

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers = None

    def fit(self, data):
        np.random.seed(0)

        # step 1 随机选取k个点为初始聚类中心点
        self.centers = np.empty((0,data.shape[1]))
        for i in np.random.choice(data.shape[0],self.k_):
            self.centers = np.append(self.centers,[data[i]],axis=0)

        # step 2 开始循环聚类
        num_iter = 0                                        # 迭代次数
        tolerance_achive = False                            # 是否已经到达误差的容差
        cluster_assment = np.zeros((data.shape[0],2))       # 保存聚类的信息矩阵，第一列为对应的聚类中心，第二维为离聚类中心的距离
        center_dist = np.zeros((data.shape[0],self.k_))     # 保存每个点到每个聚类中心的距离

        while num_iter < self.max_iter_ and (not tolerance_achive):

            for center_index in range(self.k_):
                diff = data - self.centers[center_index,:]
                diff = np.sqrt(np.sum(np.square(diff),axis=1))
                center_dist[:,center_index] = diff.reshape(data.shape[0])


            min_dist_index = np.argmin(center_dist,axis=1)
            dist = np.min(center_dist,axis=1)

            for point_index in range(data.shape[0]):
                cluster_assment[point_index,:] = min_dist_index[point_index],dist[point_index]

            tolerance_achive = True
            for center_index in range(self.k_):
                point_in_k_cluster = data[np.nonzero(cluster_assment[:,0] == center_index)[0]]
                if len(point_in_k_cluster) != 0:
                    new_mean = np.mean(point_in_k_cluster,axis=0)
                    if np.sum(np.square(self.centers[center_index,:] - new_mean)) > np.square(self.tolerance_):
                        tolerance_achive = False
                    self.centers[center_index,:] = new_mean
            num_iter += 1


    def predict(self, p_datas):
        result = []

        center_dist = np.zeros((p_datas.shape[0],self.k_))
        for center_index in range(self.k_):
            diff = p_datas - self.centers[center_index,:]
            diff = np.sqrt(np.sum(np.square(diff),axis=1))
            center_dist[:,center_index] = diff.reshape(p_datas.shape[0])
        
        result = np.argmin(center_dist,axis=1)
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)


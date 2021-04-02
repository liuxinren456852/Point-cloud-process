# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from numpy.core.fromnumeric import argmin
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from pyntcloud import PyntCloud
import pandas as pd

import open3d as o3d

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):

    # 计算每个点的法向量 利用法向量确定可能是内点的数据点
    print("开始构建KDtree")
    kdtree = KDTree(data,leaf_size=8)
    print("KDTree构建完成")

    # 计算法向量
    normals = np.empty((data.shape[0],data.shape[1]))
    bad_index = np.zeros((data.shape[0],1))

    print("开始计算每个点的法向量")
    for i in range(data.shape[0]):
        point = data[i,:]
        index = kdtree.query_radius(point.reshape(1,-1),r=1.0)
        neighbors_radius = data[index[0],:]
        if(neighbors_radius.shape[0] < 3):                                      # 认为1m内邻近点数小于3个的点没有准确的法向量
            normals[i,:] = np.zeros((1,3))
            bad_index[i]=1
        else:
            neighbors_mean = np.mean(neighbors_radius,axis=0)
            neighbors_without_mean = neighbors_radius-neighbors_mean
            H = np.dot(neighbors_without_mean.T,neighbors_without_mean)
            eigenvalues,eigenvectors = np.linalg.eig(H)
            ind = argmin(eigenvalues)
            normal = eigenvectors[:,ind].T
            normals[i,:] = normal
            if(np.sqrt(np.power(normal[0],2.0) + np.power(normal[1],2.0))/np.abs(normal[2]) > 0.57):
                bad_index[i]=1

    # 根据法向量接近垂直的个数乘以系数0.7确定RANSAC提前停止迭代的阈值
    inline_percent = (1 - bad_index.sum(axis=0)[0]/data.shape[0])*0.7

            
    print("法向量计算完成")

    # 使用RANSCA方法拟合地面
    # 地面模型:ax+by+cz+d=0

    distance_threshold = 0.5                                # 距离阈值tau
    ransac_n = 3                                            # RANSAC参数点数目
    num_iter = 50                                           # RANSAC迭代次数
    inlier_cloud = np.empty(shape=[0, data.shape[1]])       # 内点的点云

    for i in range(num_iter):
        
        print("迭代次数：%d"%(i))
        print("开始选点")
        # step1 随机选取三个点
        p = np.empty(shape=[0,data.shape[1]])
        while p.shape[0]<3:
            index = np.random.choice(data.shape[0],1)[0]
            if(bad_index[index] < 1 and  np.sqrt(np.power(normals[index][0],2.0) + np.power(normals[index][1],2.0))/np.abs(normals[index][2]) < 0.57 ):
                p = np.append(p,[data[index]],axis=0)
        print("选点结束")

        print("开始拟合平面")
        # step2 使用选取的三个点拟合平面
        a = (p[1][1] - p[0][1]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1])
        b = (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2])
        c = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0])
        d = 0 - (a * p[0][0] + b * p[0][1] + c * p[0][2])

        print("开始遍历剩余点")
        # step3 遍历剩余点，标记内点和外点
        inlier = np.empty(shape=[0, data.shape[1]])         # 内点
        inlier_idx = np.empty(shape=[0, 1], dtype=int)      # 内点序号
        for idx in range(data.shape[0]):
            p = data[idx, :]
            point_distance = abs(a*p[0] + b*p[1] + c*p[2] + d) / np.sqrt(a*a + b*b + c*c)       # 计算对应点到拟合平面的距离
            if (point_distance < distance_threshold and bad_index[idx] < 1  ):                  # 认为距离小于阈值并且法向量垂直的点为内点
                inlier = np.append(inlier, [data[idx]], axis=0)
                inlier_idx = np.append(inlier_idx, idx)

        

        if inlier.shape[0] > inlier_cloud.shape[0]:
            inlier_cloud = inlier
            segmengted_cloud = np.delete(data, inlier_idx, axis=0)

        print("内点比例:%.2f"%(inlier.shape[0]/data.shape[0]))

        if(inlier_cloud.shape[0]/data.shape[0] > inline_percent):                               # 提前停止条件
            break;

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud, inlier_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):

    distance_threshold = 1.0                               # 搜索阈值
    min_sample = 4                                         # 最小邻域点个数
    n = len(data)

    # 构建kd_tree
    leaf_size = 8
    tree = KDTree(data,leaf_size)

    #step1 初始化核心对象集合T,聚类个数k,聚类集合C, 未访问集合P
    core_sets = set()                                                        # set 集合
    k = 0                                                                    # 第k类
    cluster_index = np.zeros(n,dtype=int)                                    # 聚类集合
    unvisited = set(range(n))                                                # 初始化未访问集合

    #step2 通过判断，通过kd_tree radius NN找出所有核心点
    nearest_idx = tree.query_radius(data, distance_threshold)                # 进行radius NN搜索,半径为epsion,所有点的最临近点储存在 nearest_idx中
    for d in range(n):
        if len(nearest_idx[d]) >= min_sample:                                # 临近点数 > min_sample,加入核心点
            core_sets.add(d)                                                 # 最初的核心点

    #step3 聚类
    while len(core_sets):     
        unvisited_old = unvisited                                             # 更新为访问集合
        core = list(core_sets)[np.random.randint(0,len(core_sets))]           # 从核心点集中随机选取一个核心点core
        unvisited = unvisited - set([core])                                   # 把核心点标记为 visited,从 unvisited 集合中剔除
        visited = []
        visited.append(core)                                                  # 把核心点加入已经访问过的点

        while len(visited):
            new_core = visited[0]
            #kd-tree radius NN 搜索邻近
            if new_core in core_sets:                            # 如果当前搜索点是核心点
                S = unvisited & set(nearest_idx[new_core])       # 当前核心对象的nearest与unvisited 的交集
                visited +=  (list(S))                            # 对该newcore所能辐射的点，再做检测
                unvisited = unvisited - S                        # unvisited 剔除已visited 的点
            visited.remove(new_core)                             # newcore已做检测,去掉new core

        cluster = unvisited_old -  unvisited                     # 原有的unvisited和去掉了该核心对象的密度可达对象的visited就是该类的所有对象
        core_sets = core_sets - cluster                          # 去掉该类对象里面包含的核心对象
        cluster_index[list(cluster)] = k
        k += 1                                                   #类个数
    noise_cluster = unvisited
    cluster_index[list(noise_cluster)] = -1                      #噪声归类为-1

    return cluster_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(segmented_ground, segmented_cloud, cluster_index):
    def colormap(c, num_clusters):
        # outlier:
        if c == -1:
            color = [0]*3
        # surrouding object:
        else:
            color = [c/num_clusters*128 + 127] * 3
            color[c % 3] = 0

        return color

    # 地面的颜色
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)
    pcd_ground.colors = o3d.utility.Vector3dVector(
        [
            [0, 0, 255] for i in range(segmented_ground.shape[0])
        ]
    )

    # surrounding object elements:
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(segmented_cloud)
    num_clusters = max(cluster_index) + 1
    print(num_clusters)


    pcd_objects.colors = o3d.utility.Vector3dVector(
        [
            colormap(c, num_clusters) for c in cluster_index
        ]
    )
    o3d.visualization.draw_geometries([pcd_ground, pcd_objects])




def main():
    root_dir = 'D:/Data/Datasets/data_object_velodyne/testing/velodyne' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]


    filename = os.path.join(root_dir, cat[250])
    print("当前使用的点云文件为：")
    print('clustering pointcloud file:', filename)

    print("读取点云文件")
    origin_points = read_velodyne_bin(filename)
    print("点云大小为：")
    print(origin_points.shape)
    print("开始提取地面")
    segmented_points,ground_points = ground_segmentation(data=origin_points)
    cluster_index = clustering(segmented_points)

    plot_clusters(ground_points, segmented_points, cluster_index)

if __name__ == '__main__':
    main()

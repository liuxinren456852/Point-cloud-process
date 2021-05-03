# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct
import matplotlib.pyplot as plt

from scipy import spatial

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32).T

def main():
    # 测试配置
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    # 数据读取
    filename = "/Volumes/Data/深蓝学院课程/5-三维点云处理/Homework 2/000000.bin"
    db_np = read_velodyne_bin(filename).T

    print("octree --------------")
    construction_time = 0
    knn_time = 0
    radius_time = 0
    brute_time = 0
    
    octree_knn = []

    begin_t = time.time()
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time = time.time() - begin_t

    query = db_np[0,:]



    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    octree.octree_knn_search(root, db_np, result_set, query)
    knn_time = time.time() - begin_t

    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    octree.octree_radius_search_fast(root, db_np, result_set, query)
    radius_time = time.time() - begin_t

    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    brute_time = time.time() - begin_t

    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time*1000,
                                                                     knn_time*1000,
                                                                     radius_time*1000,
                                                                     brute_time*1000))

    for i in range(100):
        begin_t = time.time()
        result_set = KNNResultSet(capacity=i+1)
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time = time.time() - begin_t
        octree_knn.append(knn_time*1000)

  




    print("kdtree --------------")
    construction_time = 0
    knn_time = 0
    radius_time = 0
    brute_time = 0

    begin_t = time.time()
    root = kdtree.kdtree_construction(db_np, leaf_size)
    construction_time += time.time() - begin_t

    query = db_np[0,:]

    kdtree_knn = []



    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    kdtree.kdtree_knn_search(root, db_np, result_set, query)
    knn_time = time.time() - begin_t

    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    kdtree.kdtree_radius_search(root, db_np, result_set, query)
    radius_time += time.time() - begin_t

    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    brute_time = time.time() - begin_t

    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time * 1000,
                                                                     knn_time * 1000,
                                                                     radius_time * 1000,
                                                                     brute_time * 1000))   


    for i in range(100):
        begin_t = time.time()
        result_set = KNNResultSet(capacity=i+1)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time = time.time() - begin_t  
        kdtree_knn.append(knn_time*1000)



    plt.plot(range(100),octree_knn,'g',range(100),kdtree_knn)
    plt.legend(["octree","kdtree"])
    plt.xlabel("k")
    plt.ylabel("knn search time(ms)")
    plt.show()  





    print("Scipy-kdtree --------------")

    construction_time = 0
    knn_time = 0
    radius_time

    begin_t =  time.time()
    sc_tree = spatial.KDTree(db_np,leaf_size)
    construction_time = time.time() - begin_t

    begin_t =  time.time()
    sc_tree.query(query,k)
    knn_time = time.time() - begin_t
    

    print("Scipy-kdtree: build %.3f, knn %.3f" % (construction_time * 1000,
                                                  knn_time * 1000))


if __name__ == '__main__':
    main()

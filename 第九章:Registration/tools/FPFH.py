import argparse             # 命令行参数获取
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d


def SFPH(pcd,pcd_all,search_tree,key_point,radius,B):
    # 获取点云
    points = np.asarray(pcd.points)
    points_all = np.asarray(pcd_all.points)
    normals_all = np.asarray(pcd_all.normals)  

    # 获取邻居节点
    [k,idx_neighbor,_] = search_tree.search_radius_vector_3d(key_point,radius)
    
    # 获取n1
    n1 = normals_all[idx_neighbor[0]]

    if k <= 1:
        return None

    # 移除关键点本身
    idx_neighbor = idx_neighbor[1:]

    # 计算 (p2-p1)/norm(p2-p1)
    diff = points_all[idx_neighbor] - key_point
    diff = diff/np.reshape(np.linalg.norm(diff,ord=2,axis=1),(k-1,1))

    u = n1
    v = np.cross(u,diff)
    w = np.cross(u,v)    

    # 计算n2
    n2 = normals_all[idx_neighbor]

    # 计算alpha
    alpha = np.reshape((v*n2).sum(axis = 1),(k-1,1))

    # 计算phi
    phi = np.reshape((u*diff).sum(axis = 1),(k-1,1))

    # 计算 theta
    theta = np.reshape(np.arctan2((w*n2).sum(axis = 1),(u*n2).sum(axis = 1)),(k-1,1))

    # 计算相应的直方图
    alpha_hist = np.reshape(np.histogram(alpha,B,range=[-3.14,3.14])[0],(1,B))
    phi_hist = np.reshape(np.histogram(phi,B,range=[-3.14,3.14])[0],(1,B))
    theta_hist = np.reshape(np.histogram(theta,B,range=[-3.14,3.14])[0],(1,B))

 
    # 组成描述子
    fpfh = np.hstack((alpha_hist,phi_hist,theta_hist))

    return fpfh

    


# 计算FPFH描述子
def Descripter_FPFH(pcd,pcd_all,search_tree,radius,B):

    # 点云
    points = np.asarray(pcd.points)
    points_all = np.asarray(pcd_all.points)
    N, _ = points.shape

    spfh_lookup_table = {}
    description = []  

    for keypoint_id in range(N):

        key_point = np.asarray(pcd.points)[keypoint_id]

        # 寻找keypoint的邻居点
        [k,idx_neighbor,_] = search_tree.search_radius_vector_3d(key_point,radius)
        if k <= 1:
            return None
        
        # 移除关键点本身
        idx_neighbor = idx_neighbor[1:]

        # 计算权重
        w = 1.0/np.linalg.norm(key_point - points_all[idx_neighbor],ord = 2,axis = 1)

        # 计算邻居的SPFH
        neighbor_SPFH = np.reshape(np.asarray([SFPH(pcd,pcd_all,search_tree,points_all[i],radius,B) for i in idx_neighbor]),(k-1,3*B))

        neighbor_SPFH = 1.0/(k-1)*np.dot(w,neighbor_SPFH)

        spfh_keypoint = spfh_lookup_table.get(keypoint_id, None)

        if spfh_keypoint is None:
            spfh_lookup_table[keypoint_id] = SFPH(pcd,pcd_all,search_tree,key_point,radius,B)
        spfh_keypoint = spfh_lookup_table[keypoint_id]

        # 计算最终的FPFH
        fpfh = spfh_keypoint + neighbor_SPFH

        description.append(fpfh)

    description = np.asarray(description).T.reshape((3*B,N))
    return description


    

'''
Author: teamo1998
Date: 2021-05-03 15:12:57
LastEditTime: 2021-05-03 21:10:05
LastEditors: Please set LastEditors
Description: 本代码是为了完成ISS特征点检测
FilePath: /Point-cloud-process/第7章:3D Feature Detection/ISS.py
'''


import pandas as pd 
import open3d as o3d
import argparse
import numpy as np
import heapq

# 读取点云文件
def read_pointcloud(file_name):
    df = pd.read_csv(file_name,header = None)
    df.columns = ["x","y","z",
                    "nx","ny","nz"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x","y","z"]].values)
    pcd.normals = o3d.utility.Vector3dVector(df[["nx","ny","nz"]].values)

    return pcd


# 获取命令行参数
def get_args():

    parase = argparse.ArgumentParser(description="arg parser")
    parase.add_argument("--file_name",type=str,default= "/home/teamo/airplane/airplane_0001.txt")
    parase.add_argument("--radius",type=float,default= 0.1)
    parase.add_argument("--gamma32",type=float,default= 1.2)
    parase.add_argument("--gamma21",type=float,default= 1.2)
    parase.add_argument("--l3_min",type=float,default= -1.0)
    parase.add_argument("--k_min",type=int,default= 1)
    return parase.parse_args()
    



if __name__ == "__main__":

    # 记录每个点的邻居点，用于进行极大值抑制
    neighbor_every_point = []

    # 保存每个点的id lamda1，2，3
    df_data = {
        "id":[],
        "l1":[],
        "l2":[],
        "l3":[],
        "k":[]
    }

    # 获取命令行参数
    args = get_args()

    file_name = args.file_name
    radius = args.radius

    # 获取点云
    pcd = read_pointcloud(file_name)

    # 构建搜索树
    search_tree = o3d.geometry.KDTreeFlann(pcd)

    points = np.asarray(pcd.points)

    # 保存每个点的邻居点个数
    num_neighbor_cach = np.zeros((points.shape[0],1))

    # 保存每个点的lambda3
    l3 = []

    pq = []

    for idx_centor, center in enumerate(points):

        # 计算协方差矩阵

        [k,idx,_] = search_tree.search_radius_vector_3d(center,radius)

        w = []

        neighbor_every_point.append(idx)

        for index in idx:
            if(num_neighbor_cach[index] == 0):
                [k_,_,_] = search_tree.search_radius_vector_3d(points[index],radius)
                num_neighbor_cach[index] = k_
            
            w.append(num_neighbor_cach[index])

        neighbor = points[idx]

        distance = neighbor - center

        w = np.asarray(w)

        w = np.reshape(w,(-1,))
        
        # 协方差矩阵
        cov = 1.0/w.sum()*np.dot(distance.T,np.dot(np.diag(w),distance))

        eigen_values,_ = np.linalg.eig(cov)

        eigen_values = eigen_values[np.argsort(eigen_values)[::-1]]

        l3.append(eigen_values[-1])
        
        
        # 构造小顶堆，用于非极大值抑制
        heapq.heappush(pq,(-eigen_values[2],idx_centor))

        df_data["id"].append(idx_centor)
        df_data["l1"].append(eigen_values[0])
        df_data["l2"].append(eigen_values[1])
        df_data["l3"].append(eigen_values[2])
        df_data["k"].append(k)

    # 获取l3的阈值
    l3_min = 2.0*np.mean(np.asarray(l3),axis=0)
    if(args.l3_min > 0.0):
        l3_min = args.l3_min


    print(l3_min)
    
    suppressed = set()

    # 非极大值抑制
    while(pq):
        _,idx_centor = heapq.heappop(pq)
        if not idx_centor in suppressed:
            neighbor = neighbor_every_point[idx_centor]

            neighbor = neighbor[1:]

            for _i in neighbor:
                suppressed.add(_i)
            

        else:
            continue


    df_data = pd.DataFrame.from_dict(df_data)

    # 排除非极大值点
    df_data = df_data.loc[df_data["id"].apply(lambda id:not id in suppressed),df_data.columns]

    # 排除不符合协方差要求的点
    df_data = df_data.loc[( df_data["l1"] > df_data["l2"]*args.gamma21) & ( df_data["l2"] > df_data["l3"]*args.gamma32) & (df_data["l3"] > l3_min) & (df_data["k"] > args.k_min) ,df_data.columns]


    pcd.paint_uniform_color([0.95,0.95,0.95])

    np.asarray(pcd.colors)[df_data["id"].values,:] = [1.0,0.0,0.0]

    
    o3d.visualization.draw_geometries([pcd])


# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import pandas as pd
import math
from pyntcloud import PyntCloud
import random

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []

    point_cloud = np.array(point_cloud, dtype=np.float64)
    # 分离三个轴的数据
    x_array = point_cloud[:, 0]
    y_array = point_cloud[:, 1]
    z_array = point_cloud[:, 2]
    # 获取三个轴的最大值和最小值
    x_max, x_min = np.max(x_array), np.min(x_array)
    y_max, y_min = np.max(y_array), np.min(y_array)
    z_max, z_min = np.max(z_array), np.min(z_array)

    # 计算网格维度
    D_x = math.ceil((x_max - x_min) / leaf_size)
    D_y = math.ceil((y_max - y_min) / leaf_size)
    D_z = math.ceil((z_max - z_min) / leaf_size)

    # 分别计算每个点的所属网格
    h_array = np.array([], dtype=np.float64)
    for point in point_cloud:
        point = np.array(point, dtype=np.float64)
        h_x = np.floor((point[0] - x_min) / leaf_size)
        h_y = np.floor((point[1] - y_min) / leaf_size)
        h_z = np.floor((point[2] - z_min) / leaf_size)
        h = h_x + h_y * D_x + h_z * D_x * D_y
        h_array = np.append(h_array, h)

    sort = h_array.argsort()

    point_cloud = point_cloud[sort]

    # 在每个网格中选取一个点
    voxel = []
    for i in range(len(point_cloud)):
        # print(i)
        if i < len(point_cloud) - 1 and h_array[sort][i] == h_array[sort][i+1]:
            voxel.append(i)
        else:
            voxel.append(i)
            # random
            #random_index = random.randint(0,len(voxel)-1)
            #choice_point = point_cloud[voxel[random_index],:]
            # centroid
            choice_point = np.mean(point_cloud[i-len(voxel)+1:i+1, :], axis=0)

            filtered_points.append(choice_point)
            voxel = []


    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print(filtered_points.shape)
    return filtered_points

def main():

    # 加载txt格式原始点云
    points = pd.read_csv("D:/Data/Datasets/modelnet40_normal_resampled/airplane/airplane_0001.txt")
    points = points.iloc[:,0:3]
    points.columns = ["x","y","z"]
    point_cloud_pynt = PyntCloud(points)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.08)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    #显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()

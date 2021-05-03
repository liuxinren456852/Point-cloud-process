'''
Author: teamo1998
Date: 2021-05-03 00:49:49
LastEditTime: 2021-05-03 13:35:07
LastEditors: Please set LastEditors
Description: 本代码重要是为了熟悉global registration的流程，使用了open3d的API构建流程
FilePath: /Homework 9/Registration_o3d.py
'''


import os
import argparse
import progressbar

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation
import copy
'''
@description: 获取命令行参数
@param {*}
@return {*}
'''
def get_args():
    parase = argparse.ArgumentParser("Registration")
    parase.add_argument("--pointcloud_dir",type=str,default="/media/teamo/samsung/Homework 9/registration_dataset")
    parase.add_argument("--voxel_size",type=float,default=1)


    return parase.parse_args()


'''
@description: 初始化输出模板，使用与提供的结果相同的格式利用pd输出
@param {*}
@return {*}
'''
def init_output():
    df_output = {
        'idx1': [],
        'idx2': [],
        't_x': [],
        't_y': [],
        't_z': [],
        'q_w': [],
        'q_x': [],
        'q_y': [],
        'q_z': []
    }

    return df_output

'''
@description: 绘制配准之后的点云
@param {*} source
@param {*} target
@param {*} transformation
@return {*}
'''
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

'''
@description:  以pandas的dataframe格式读取结果文件
@param {*} results_path
@return {*} pandas.DataFrame
'''
def read_registration_results(results_path):

    df_results = pd.read_csv(
        results_path
    )

    return df_results

'''
@description: 读取bin格式的点云
@param {*} bin_path
@return {*} open3d.pcd
'''
def read_point_cloud_bin(bin_path):

    data = np.fromfile(bin_path, dtype=np.float32)

    # 将数据重新格式化
    N, D = data.shape[0]// 6, 6
    point_cloud_with_normal = np.reshape(data, (N, D))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 0:3])
    point_cloud.normals = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 3:6])

    return point_cloud

'''
@description: 用于向结果DataFrame中加入数据
@param {*} df_output
@param {*} idx1
@param {*} idx2
@param {*} T
@return {*}
'''
def add_to_output(df_output, idx1, idx2, T):
    """
    Add record to output
    """
    def format_transform_matrix(T):
        r = Rotation.from_matrix(T[:3, :3])
        q = r.as_quat()
        t = T[:3, 3]

        return (t, q)

    df_output['idx1'].append(idx1)
    df_output['idx2'].append(idx2)
    
    (t, q) = format_transform_matrix(T)

    # translation:
    df_output['t_x'].append(t[0])
    df_output['t_y'].append(t[1])
    df_output['t_z'].append(t[2])
    # rotation:
    df_output['q_w'].append(q[3])
    df_output['q_x'].append(q[0])
    df_output['q_y'].append(q[1])
    df_output['q_z'].append(q[2])

'''
@description: 将结果保存为txt文件
@param {*} filename
@param {*} df_output
@return {*}
'''
def write_output(filename, df_output):
    df_output = pd.DataFrame.from_dict(
        df_output
    )

    print(f'write output to {filename}')
    df_output[
        [
            'idx1', 'idx2',
            't_x', 't_y', 't_z',
            'q_w', 'q_x', 'q_y', 'q_z'
        ]
    ].to_csv(filename, index=False)

if __name__ == "__main__":

    args = get_args()
    datasets_dir = args.pointcloud_dir
    
    # 进度条
    progress = progressbar.ProgressBar()


    # 结果dataframe
    registration_results = read_registration_results(os.path.join(datasets_dir,"reg_result.txt"))

    # 初始化输出文件结构
    df_output = init_output()

    '''
    @description: 迭代结果文件中的每一行，获取对应的点云
    @param {*}
    @return {*}
    '''
    voxel_size = args.voxel_size

    for index,row in progress(list(registration_results.iterrows())):
        
        # 需要配准的点云id
        idx_target = int(row["idx1"])
        idx_source = int(row["idx2"])
        
        # 读取点云,输出格式为open3d的点云格式
        pcd_source = read_point_cloud_bin(os.path.join(datasets_dir,"point_clouds",f"{idx_source}.bin"))
        pcd_target = read_point_cloud_bin(os.path.join(datasets_dir,"point_clouds",f"{idx_target}.bin"))
        draw_registration_result(pcd_source,pcd_target,np.identity(4))

        pcd_down_source = pcd_source.voxel_down_sample(voxel_size)
        pcd_down_target = pcd_target.voxel_down_sample(voxel_size)

        # 绘制初始结果
        draw_registration_result(pcd_down_source,pcd_down_target,np.identity(4))

        # 降采样，这里的降采样对于提高RANSAC的效果十分明显
        radius_feature = voxel_size * 5
        pcd_fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down_source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        pcd_fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down_target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        distance_threshold = voxel_size * 1.5

        # 使用open3d的RANSAC流程函数进行global registration
        
        init_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_down_source, pcd_down_target, pcd_fpfh_source, pcd_fpfh_target, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
        # 这三个参数用来评价RANSAC配准的结果是否有效
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(              # 这个参数用来检测配准之后的两个点云的几何不变性       
                0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(                # 这个用于检测配准之后两个点云的距离接近性
                distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(10*2*3.14/180)    # 这个用于检测配准后两个点云的法向量相似性
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))             # RANSAC的最大迭代次数及置信概率

        #  绘制配准后的结果
        draw_registration_result(pcd_down_source, pcd_down_target, init_result.transformation)

        distance_threshold = 0.05 * 0.4

        # 使用ICP优化结果 这里由于已经有了初始值，选择使用原始点云
        final_result = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, distance_threshold, init_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

        draw_registration_result(pcd_source, pcd_target, final_result.transformation)

        # 往结果DataFrame中加入数据
        add_to_output(df_output, idx_target, idx_source, final_result.transformation)
    
    write_output(
        os.path.join(datasets_dir, 'reg_result_teamo.txt'),
        df_output)
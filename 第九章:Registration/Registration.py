'''
Author: your name
Date: 2021-05-02 20:25:29
LastEditTime: 2021-05-03 16:37:16
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Homework 9/Registration.py
'''
'''
@description: 
@param {*}
@return {*}
'''


import os
import argparse
import progressbar

import numpy as np
import open3d as o3d
import pandas as pd
import copy

from tools import IO
from tools import ISS
from tools import FPFH
from tools import RANSAC


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

def get_args():
    parase = argparse.ArgumentParser("Registration")
    parase.add_argument("--pointcloud_dir",type=str,default="/Volumes/Data/深蓝学院课程/5-三维点云处理/Point-cloud-process/第九章:Registration/registration_dataset")
    parase.add_argument("--radius",type=float,default=0.5)


    return parase.parse_args()

if __name__ == "__main__":

    args = get_args()
    datasets_dir = args.pointcloud_dir
    radius = args.radius
    
    # 进度条

    progress = progressbar.ProgressBar()
    # 结果dataframe
    registration_results = pd.DataFrame(IO.read_registration_results(os.path.join(datasets_dir,"reg_result.txt")))

    df_output = IO.init_output()
    for index,row in progress(list(registration_results.iterrows())):
        
        idx_source = int(row["idx2"])
        idx_target = int(row["idx1"])

        # 读取点云,输出格式为open3d的点云格式
        pcd_source = IO.read_point_cloud_bin(os.path.join(datasets_dir,"point_clouds",f"{idx_source}.bin"))
        pcd_target = IO.read_point_cloud_bin(os.path.join(datasets_dir,"point_clouds",f"{idx_target}.bin"))

        # 移除指定范围内没有邻居的外点
        pcd_source, ind = pcd_source.remove_radius_outlier(nb_points=4, radius=radius)
        pcd_target, ind = pcd_target.remove_radius_outlier(nb_points=4, radius=radius)

        # 构建kdtree
        search_tree_source = o3d.geometry.KDTreeFlann(pcd_source)
        search_tree_target = o3d.geometry.KDTreeFlann(pcd_target)

        # 特征点检测
        keypoints_source = ISS.detect(pcd_source,search_tree_source,radius)
        keypoints_target = ISS.detect(pcd_target,search_tree_target,radius)

        # 关键点
        pcd_target_keypoints = pcd_target.select_by_index(keypoints_target['id'].values)
        pcd_source_keypoints = pcd_source.select_by_index(keypoints_source['id'].values)

        # 提取描述子 此处由于自己实现的描述子计算在python环境下运行太慢，故以open3d的算法代替,结果是一样的
        # fpfh_source_keypoints = FPFH.Descripter_FPFH(pcd_source_keypoints,pcd_source,search_tree_source,radius,11)
        # fpfh_target_keypoints = FPFH.Descripter_FPFH(pcd_target_keypoints,pcd_target,search_tree_target,radius,11)

        fpfh_source_keypoints = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_source_keypoints, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=5*radius, max_nn=100)
        ).data
        
        fpfh_target_keypoints = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_target_keypoints, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=5*radius, max_nn=100)
        ).data

        distance_threshold_init = 1.5 * radius
        distance_threshold_final = 1.0 * radius
        
        # RANSAC初始匹配
        init_result = RANSAC.ransac_match(
            pcd_source_keypoints, pcd_target_keypoints, 
            fpfh_source_keypoints, fpfh_target_keypoints,    
            ransac_params = RANSAC.RANSACParams(
                max_workers=5,
                num_samples=4, 
                max_correspondence_distance=distance_threshold_init,
                max_iteration=200000, 
                max_validation=500,
                max_refinement=30
            ),
            checker_params = RANSAC.CheckerParams(
                max_correspondence_distance=distance_threshold_init,
                max_edge_length_ratio=0.9,
                normal_angle_threshold=None
            )      
        )
        
        final_result = RANSAC.exact_match(
            pcd_source, pcd_target, search_tree_target,
            init_result.transformation,
            distance_threshold_final, 60
        )

        IO.add_to_output(df_output, idx_target, idx_source, final_result.transformation)

    IO.write_output(
        os.path.join(datasets_dir, 'reg_result_teamo1998_test.txt'),
        df_output
    )
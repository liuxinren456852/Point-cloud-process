import collections
import copy
import concurrent.futures

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R
import open3d as o3d

'''
@description: RANSAC 匹配参数:
@param {*}
@return {*}
'''
RANSACParams = collections.namedtuple(
    'RANSACParams',
    [
        'max_workers',
        'num_samples', 
        'max_correspondence_distance', 'max_iteration', 'max_validation', 'max_refinement'
    ]
)

CheckerParams = collections.namedtuple(
    'CheckerParams', 
    ['max_correspondence_distance', 'max_edge_length_ratio', 'normal_angle_threshold']
)


'''
@description: 根据相应的描述子计算匹配对
@param {*}
@return {*}
'''
def get_potential_matches(feature_source, feature_target):

    # 在高维空间构建对应的搜索树
    search_tree = o3d.geometry.KDTreeFlann(feature_target)

    # 为原始点云中的每一个点寻找对应的特征点
    _, N = feature_source.shape
    matches = []
    for i in range(N):
        query = feature_source[:, i]
        _, idx_nn_target, _ = search_tree.search_knn_vector_xd(query, 1)
        matches.append(
            [i, idx_nn_target[0]]
        )
    
    # 结果为N*2的数组
    matches = np.asarray(
        matches
    )

    return matches

'''
@description: 闭式求解ICP 
@param {*} source  原始点云
@param {*} target  目标点云
@return {*} T SE3
'''
def solve_icp(source, target):
    # 计算均值:
    up = source.mean(axis = 0)
    uq = target.mean(axis = 0)

    # 去重心化后的点云:
    P_centered = source - up
    Q_centered = target - uq

    # SVD分解求R和T
    U, s, V = np.linalg.svd(np.dot(Q_centered.T, P_centered), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    t = uq - np.dot(R, up)

    # 将R和T变换为变换矩阵的格式
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.0

    return T

'''
@description: 用来检测给定给的匹配对在给定参数下是否有效 
@param {*}
@return {*}
'''
def is_valid_match(
    pcd_source, pcd_target,
    proposal,
    checker_params 
):
    """
    Check proposal validity using the fast pruning algorithm
    Parameters
    ----------
    pcd_source: open3d.geometry.PointCloud
        source point cloud
    pcd_target: open3d.geometry.PointCloud
        target point cloud
    proposal: numpy.ndarray
        RANSAC potential as num_samples-by-2 numpy.ndarray
    checker_params:
        fast pruning algorithm configuration
    Returns
    ----------
    T: transform matrix as numpy.ndarray or None
        whether the proposal is a valid match for validation
    """
    idx_source, idx_target = proposal[:,0], proposal[:,1]

    # TODO: this checker should only be used for pure translation

    # 法向量方向检查
    if not checker_params.normal_angle_threshold is None:
        # get corresponding normals:
        normals_source = np.asarray(pcd_source.normals)[idx_source]
        normals_target = np.asarray(pcd_target.normals)[idx_target]

        # a. normal direction check:
        normal_cos_distances = (normals_source*normals_target).sum(axis = 1)
        is_valid_normal_match = np.all(normal_cos_distances >= np.cos(checker_params.normal_angle_threshold)) 

        if not is_valid_normal_match:
            return None

    # 获取相关点
    points_source = np.asarray(pcd_source.points)[idx_source]
    points_target = np.asarray(pcd_target.points)[idx_target]

    # 几何相似性检查:
    pdist_source = pdist(points_source)
    pdist_target = pdist(points_target)
    is_valid_edge_length = np.all(
        np.logical_and(
            pdist_source > checker_params.max_edge_length_ratio * pdist_target,
            pdist_target > checker_params.max_edge_length_ratio * pdist_source
        )
    )

    if not is_valid_edge_length:
        return None

    # 相关距离检查
    T = solve_icp(points_source, points_target)
    R, t = T[0:3, 0:3], T[0:3, 3]
    deviation = np.linalg.norm(
        points_target - np.dot(points_source, R.T) - t,
        axis = 1
    )
    is_valid_correspondence_distance = np.all(deviation <= checker_params.max_correspondence_distance)

    return T if is_valid_correspondence_distance else None

def shall_terminate(result_curr, result_prev):
    # relative fitness improvement:
    relative_fitness_gain = result_curr.fitness / result_prev.fitness - 1

    return relative_fitness_gain < 0.01


def exact_match(
    pcd_source, pcd_target, search_tree_target,
    T,
    max_correspondence_distance, max_iteration
):
    """
    Perform exact match on given point cloud pair
    Parameters
    ----------
    pcd_source: open3d.geometry.PointCloud
        source point cloud
    pcd_target: open3d.geometry.PointCloud
        target point cloud
    search_tree_target: scipy.spatial.KDTree
        target point cloud search tree
    T: numpy.ndarray
        transform matrix as 4-by-4 numpy.ndarray
    max_correspondence_distance: float
        correspondence pair distance threshold
    max_iteration:
        max num. of iterations 
    Returns
    ----------
    result: open3d.registration.RegistrationResult
        Open3D registration result
    """
    # num. points in the source:
    N = len(pcd_source.points)

    # evaluate relative change for early stopping:
    result_prev = result_curr = o3d.pipelines.registration.evaluate_registration(
        pcd_source, pcd_target, max_correspondence_distance, T
    )

    for _ in range(max_iteration):
        # TODO: transform is actually an in-place operation. deep copy first otherwise the result will be WRONG
        pcd_source_current = copy.deepcopy(pcd_source)
        # apply transform:
        pcd_source_current = pcd_source_current.transform(T)
        
        # find correspondence:
        matches = []
        for n in range(N):
            query = np.asarray(pcd_source_current.points)[n]
            _, idx_nn_target, dis_nn_target = search_tree_target.search_knn_vector_3d(query, 1)

            if dis_nn_target[0] <= max_correspondence_distance:
                matches.append(
                    [n, idx_nn_target[0]]
                )
        matches = np.asarray(matches)

        if len(matches) >= 4:
            # sovle ICP:
            P = np.asarray(pcd_source.points)[matches[:,0]]
            Q = np.asarray(pcd_target.points)[matches[:,1]]
            T = solve_icp(P, Q)

            # evaluate:
            result_curr = o3d.pipelines.registration.evaluate_registration(
                pcd_source, pcd_target, max_correspondence_distance, T
            )

            # if no significant improvement:
            if shall_terminate(result_curr, result_prev):
                print('[RANSAC ICP]: Early stopping.')
                break

    return result_curr

'''
@description: RANSAC匹配主函数
@param {*}
@return {*}
'''
def ransac_match(
    pcd_source, pcd_target, 
    feature_source, feature_target,
    ransac_params, checker_params
):

    # 获取对应的特征空间匹配对:
    matches = get_potential_matches(feature_source, feature_target)

    #print(matches)
    # 为目标点云构建搜索树:
    search_tree_target = o3d.geometry.KDTreeFlann(pcd_target)

    # RANSAC:
    N, _ = matches.shape
    idx_matches = np.arange(N)

    # SE3
    T = None
    
    # 构造一个生成器，用来随机选取四个点
    proposal_generator = (
        matches[np.random.choice(idx_matches, ransac_params.num_samples, replace=False)] for _ in iter(int, 1)
    )

    # 验证器，用来根据给定的检查参数获取符合要求的匹配对:
    validator = lambda proposal: is_valid_match(pcd_source, pcd_target, proposal, checker_params)

    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=ransac_params.max_workers) as executor:            
        for T in map(
            validator, 
            proposal_generator
        ):  
            if not (T is None):
                break

    # set baseline:
    # 得到一个初始的最好结果，用于RANSAC迭代的判断
    print('[RANSAC ICP]: Get first valid proposal. Start registration...')
    best_result = exact_match(
        pcd_source, pcd_target, search_tree_target,
        T,
        ransac_params.max_correspondence_distance, 
        ransac_params.max_refinement
    )



    # RANSAC:
    num_validation = 0
    for i in range(ransac_params.max_iteration):
        # get proposal:
        T = validator(next(proposal_generator))

        # check validity:
        if (not (T is None)) and (num_validation < ransac_params.max_validation):
            num_validation += 1

            # refine estimation on all keypoints:
            result = exact_match(
                pcd_source, pcd_target, search_tree_target,
                T,
                ransac_params.max_correspondence_distance, 
                ransac_params.max_refinement
            )
            
            # update best result:
            best_result = best_result if best_result.fitness > result.fitness else result

            if num_validation == ransac_params.max_validation:
                break

    return best_result
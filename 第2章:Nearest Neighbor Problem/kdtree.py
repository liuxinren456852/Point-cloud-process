# kdtree的具体实现，包括构建和查找

import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet

# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted

# 功能：随机分割轴
# 输入：
#     axis：当前的分割轴
#     dim：总维度
def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1

# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar：初始分割轴
#     leaf_size: scalar：叶子节点包含的最多数据点
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)  #构建节点

    # 决定是否分割当前节点
    if len(point_indices) > leaf_size:
        # 获取分割位置
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # 按给定的轴对于点进行排序
        
        # 位于分割轴左边的最远一个点的索引
        middle_left_index = math.ceil(point_indices_sorted.shape[0]/2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_index]
        middle_left_point_value = db[middle_left_point_idx,axis]

        # 位于分割轴右边的最近一个点的索引
        middle_right_index = middle_left_index +1
        middle_right_point_idx = point_indices_sorted[middle_right_index]
        middle_right_point_value = db[middle_right_point_idx,axis]

        # 当前节点的值 = 距离分割值最近的两个点的平均
        root.value = (middle_left_point_value + middle_right_point_value) * 0.5

        # 递归构建左边和右边叶子节点
        root.left = kdtree_recursive_build(root.left,db,point_indices_sorted[0:middle_right_index],axis_chosed_by_cov(db[point_indices_sorted[0:middle_right_index]]),leaf_size)
        root.right = kdtree_recursive_build(root.right,db,point_indices_sorted[middle_right_index:],axis_chosed_by_cov(db[point_indices_sorted[middle_right_index:]]),leaf_size)

    return root
# 功能：按照协方差的大小选取分割轴
# 输入：
#     data: 需要分割的数据点
def axis_chosed_by_cov(data):
    axis = 0
    max_cov = 0
    for i in range(data.shape[1]):
        if np.cov(data[:,i]) > max_cov:
            max_cov = np.cov(data[:,i])
            axis = i
    return axis

# 功能：遍历一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据 N*3维
#     leaf_size：scale :叶子节点尺寸
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # 比较当前叶子节点下的每一个数据点
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 判断先从分割轴的哪一侧开始搜索
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left,db,result_set,query)
        # 判断是否需要搜索另外一侧
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right,db,result_set,query)
    else:
        kdtree_knn_search(root.right,db,result_set,query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left,db,result_set,query)

    return False

# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # 遍历叶节点下的点
    if root.is_leaf():
        # 比较当前叶子节点下的每一个数据点
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    # 判断先从分割轴的哪一侧开始搜索
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left,db,result_set,query)
        # 判断是否需要搜索另外一侧
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right,db,result_set,query)
    else:
        kdtree_radius_search(root.right,db,result_set,query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left,db,result_set,query)

    return False




def main():
    # 测试集配置
    db_size = 64                        # 数据集中的数据点个数
    dim = 3                             # 数据点的维度

    leaf_size = 4                       #叶子节点包含的最多数据点个数
    k = 1                               #搜索个数

    db_np = np.random.rand(db_size, dim) #N*3维的随机数据点

    root = kdtree_construction(db_np, leaf_size=leaf_size)  #构建kd树

    # 遍历当前树，获取每一个叶子节点的信息
    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    # Knn搜索测试
    query = np.asarray([0, 0, 0])        
    result_set = KNNResultSet(capacity=k)
    kdtree_knn_search(root, db_np, result_set, query)
    #
    print(result_set)
    
    # 暴力搜索测试
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])

    # Radiusnn搜索测试
    print("Radius search:")
    query = np.asarray([0, 0, 0])
    result_set = RadiusNNResultSet(radius = 0.5)
    kdtree_radius_search(root, db_np, result_set, query)
    print(result_set)


if __name__ == '__main__':
    main()
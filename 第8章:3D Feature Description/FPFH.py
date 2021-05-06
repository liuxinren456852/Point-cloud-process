import argparse             # 命令行参数获取
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d


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
    parase.add_argument("--file_name",type=str,default= "/home/teamo/chair/chair_0001.txt")               # 需要提取描述子的的点云文件路径
    parase.add_argument("--radius",type=float,default= 0.05)                                             # PFPH需要的参数
    parase.add_argument("--x",type=float,default=0.084)                                                # 特征点的x坐标
    parase.add_argument("--y",type=float,default=0.2597)                                               # 特征点的y坐标
    parase.add_argument("--z",type=float,default=-0.0713)                                               # 特征点的z坐标
    parase.add_argument("--B",type=int,default=11)                                                      # 描述子的分段数

    return parase.parse_args()

# 获取每个节点的SFPH描述子
def SFPH(pcd,search_tree,key_point,radius,B):
    # 获取点云
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)  

    # 获取邻居节点
    [k,idx_neighbor,_] = search_tree.search_radius_vector_3d(key_point,radius)
    
    # 获取n1
    n1 = normals[idx_neighbor[0]]

    # 移除关键点本身
    idx_neighbor = idx_neighbor[1:]

    # 计算 (p2-p1)/norm(p2-p1)
    diff = points[idx_neighbor] - key_point
    diff = diff/np.reshape(np.linalg.norm(diff,ord=2,axis=1),(k-1,1))

    u = n1
    v = np.cross(u,diff)
    w = np.cross(u,v)    

    # 计算n2
    n2 = normals[idx_neighbor]

    # 计算alpha
    alpha = np.reshape((v*n2).sum(axis = 1),(k-1,1))

    # 计算phi
    phi = np.reshape((u*diff).sum(axis = 1),(k-1,1))

    # 计算 theta
    theta = np.reshape(np.arctan2((w*n2).sum(axis = 1),(u*n2).sum(axis = 1)),(k-1,1))

    # 计算相应的直方图
    alpha_hist = np.reshape(np.histogram(alpha,B,range=[-1.0,1.0])[0],(1,B))
    phi_hist = np.reshape(np.histogram(phi,B,range=[-1.0,1.0])[0],(1,B))
    theta_hist = np.reshape(np.histogram(theta,B,range=[-3.14,3.14])[0],(1,B))

 
    # 组成描述子
    fpfh = np.hstack((alpha_hist,phi_hist,theta_hist))

    return fpfh

    


# 计算FPFH描述子
def Descripter_FPFH(pcd,search_tree,key_point,radius,B):

    # 点云
    points = np.asarray(pcd.points)
                           

    # 寻找keypoint的邻居点
    [k,idx_neighbor,_] = search_tree.search_radius_vector_3d(key_point,radius)
    if k <= 1:
        return None
    
    # 移除关键点本身
    idx_neighbor = idx_neighbor[1:]

    # 计算权重
    w = 1.0/np.linalg.norm(key_point - points[idx_neighbor],ord = 2,axis = 1)

    # 计算邻居的SPFH
    neighbor_SPFH = np.reshape(np.asarray([SFPH(pcd,search_tree,points[i],radius,B) for i in idx_neighbor]),(k-1,3*B))

    # 计算自身的描述子
    self_SFPH = SFPH(pcd,search_tree,key_point,radius,B)

    # 计算最终的FPFH
    neighbor_SPFH = 1.0/(k-1)*np.dot(w,neighbor_SPFH)

    fpfh = self_SFPH + neighbor_SPFH

    return fpfh


if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    radius = args.radius
    file_name = args.file_name
    B = args.B
    
    # 读取点云
    pcd = read_pointcloud(file_name)

    # 构建搜索树
    search_tree = o3d.geometry.KDTreeFlann(pcd)

    # 测试描述子

    key_point1 = np.array([0.4333,-0.7807,-0.4372])
    key_point2 = np.array([-0.4240,-0.7850,-0.4392])
    key_point3 = np.array([0.02323,0.004715,-0.2731])

    fpfh1 = Descripter_FPFH(pcd,search_tree,key_point1,radius,B)
    fpfh2 = Descripter_FPFH(pcd,search_tree,key_point2,radius,B)
    fpfh3 = Descripter_FPFH(pcd,search_tree,key_point3,radius,B)

    fpfh1 = fpfh1/np.linalg.norm(fpfh1)
    fpfh2 = fpfh2/np.linalg.norm(fpfh2)
    fpfh3 = fpfh3/np.linalg.norm(fpfh3)

    # print(fpfh)

    plt.plot(range(3*B), fpfh1.T, ls="-.",color="r",marker =",", lw=2, label="keypoint1")
    plt.plot(range(3*B), fpfh2.T, ls="-.",color="g",marker =",", lw=2, label="keypoint2")
    plt.plot(range(3*B), fpfh3.T, ls="-.",color="b",marker =",", lw=2, label="keypoint3")

    plt.legend()

    plt.show()


    

import pandas as pd 
import open3d as o3d
import numpy as np
import heapq



def detect(point_cloud,search_tree,radius):

    # 记录每个点的邻居点，用于进行极大值抑制
    neighbor_every_point = []

    # 保存每个点的id lamda1，2，3
    df_data = {
        "id":[],
        "x":[],
        "y":[],
        "z":[],
        "l1":[],
        "l2":[],
        "l3":[],
        "k":[]
    }

    points = np.asarray(point_cloud.points)

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
        df_data['x'].append(center[0])
        df_data['y'].append(center[1])
        df_data['z'].append(center[2])
        df_data["l1"].append(eigen_values[0])
        df_data["l2"].append(eigen_values[1])
        df_data["l3"].append(eigen_values[2])
        df_data["k"].append(k)

    
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
    df_data = df_data.loc[( df_data["l1"] > df_data["l2"]) & ( df_data["l2"] > df_data["l3"])  ,df_data.columns]

    keypoints = df_data.sort_values('l3', axis=0, ascending=False, ignore_index=True)

    return keypoints
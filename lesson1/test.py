import open3d as o3d 
import numpy as np

def main():
    np_pc = np.random.random((1000,3))
    pc_view = o3d.geometry.PointCloud()
    pc_view.points = o3d.utility.Vector3dVector(np_pc)
    o3d.visualization.draw_geometries([pc_view])

if __name__ == "__main__":
    main()
import open3d as o3d
import numpy as np


def estimate_point_spacing(pcd, k=6):
    """估计点云平均点距"""
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    points = np.asarray(pcd.points)
    for i in range(0, len(points), max(1, len(points)//5000)):  # 采样计算加速
        [_, idx, dist] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        dists.append(np.mean(np.sqrt(dist[1:])))
    return np.mean(dists)


def simulate_z_axis_scan_auto(input_ply, output_ply):
    """从 CAD 点云沿 Z 轴从上往下模拟扫描，自动匹配原始点云密度"""
    print(f" 读取点云: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)

    # 自动估计原点云密度
    spacing = estimate_point_spacing(pcd)
    print(f" 平均点距估计: {spacing:.4f}")

    # 投影分辨率设为平均点距
    resolution = spacing

    points = np.asarray(pcd.points)
    print(f"原始点数: {len(points)}")

    # 构建 XY 投影网格
    xy_min = points[:, :2].min(axis=0)
    xy_max = points[:, :2].max(axis=0)
    grid_size = np.ceil((xy_max - xy_min) / resolution).astype(int)

    z_map = np.full((grid_size[0] + 1, grid_size[1] + 1), -np.inf)
    index_map = np.full((grid_size[0] + 1, grid_size[1] + 1), -1, dtype=int)
    grid_xy = np.floor((points[:, :2] - xy_min) / resolution).astype(int)

    for i, (gx, gy) in enumerate(grid_xy):
        if points[i, 2] > z_map[gx, gy]:
            z_map[gx, gy] = points[i, 2]
            index_map[gx, gy] = i

    visible_indices = index_map[index_map >= 0]
    visible_points = points[visible_indices]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(visible_points)
    if pcd.has_colors():
        new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[visible_indices])

    o3d.io.write_point_cloud(output_ply, new_pcd)
    print(f"✅ 虚拟扫描点云已保存: {output_ply}")
    print(f"新点数: {len(visible_points)}")

    o3d.visualization.draw_geometries([new_pcd],
                                      window_name='Virtual Line-Scan (Z↓)',
                                      width=1000, height=800)


if __name__ == "__main__":
    simulate_z_axis_scan_auto("27_meshsample.ply", "virtual_scan_z.ply")

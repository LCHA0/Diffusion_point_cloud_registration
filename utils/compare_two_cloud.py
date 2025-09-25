import open3d as o3d

# 读取两个点云
src_pcd = o3d.io.read_point_cloud("wob3000.ply")     # 扫描点云
model_pcd = o3d.io.read_point_cloud("X2backed.ply") # CAD 转出来的点云

# 给点云上颜色（便于区分）
src_pcd.paint_uniform_color([1, 0, 0])     # 红色
model_pcd.paint_uniform_color([0, 1, 0])   # 绿色

# 可视化
o3d.visualization.draw_geometries([src_pcd, model_pcd])

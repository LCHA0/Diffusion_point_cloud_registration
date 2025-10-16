import open3d as o3d
import numpy as np
import ast  # 用于安全地解析txt中的Python格式数组


def load_transform_matrix(file_path):
    """
    从txt文件读取4x4变换矩阵（格式如 [ [a,b,c,d], ... ] ）
    """
    with open(file_path, "r") as f:
        text = f.read().strip()
    try:
        matrix = np.array(ast.literal_eval(text), dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"文件中矩阵形状错误: {matrix.shape}，应为 (4,4)")
        print(f"✅ 已读取变换矩阵 ({file_path}):\n{matrix}")
        return matrix
    except Exception as e:
        raise ValueError(f"读取或解析矩阵失败: {e}")


def apply_transform_to_pcd(pcd, transform_matrix):
    """将转换矩阵应用到点云"""
    return pcd.transform(transform_matrix)


def evaluate_registration_error(transformed_pcd, target_pcd):
    """计算配准误差并返回详细统计（单位：mm）"""
    distances = np.asarray(transformed_pcd.compute_point_cloud_distance(target_pcd))
    mean_error = np.mean(distances)
    rmse = np.sqrt(np.mean(distances ** 2))
    max_error = np.max(distances)
    print("\n 配准误差统计（单位：mm）")
    print(f"   平均偏差: {mean_error:.3f} mm")
    print(f"   均方根误差 (RMSE): {rmse:.3f} mm")
    print(f"   最大偏差: {max_error:.3f} mm")
    return mean_error, rmse, max_error, distances


def plt_colormap(values):
    """使用 matplotlib 生成距离误差的伪彩色映射"""
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    norm = colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    colormap = cm.get_cmap('jet')
    return colormap(norm(values))[:, :3]


def visualize_registration(source_pcd, target_pcd, transformed_pcd, distances=None):
    """可视化源点云、目标点云及误差热力图"""
    target_pcd.paint_uniform_color([0, 1, 0])  # 绿色：目标点云
    source_pcd.paint_uniform_color([1, 0, 0])  # 红色：原始源点云

    if distances is not None:
        distances_clipped = np.clip(distances, 0, np.percentile(distances, 95))
        colors = plt_colormap(distances_clipped)
        transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
        print(" 已将误差大小映射为颜色（蓝=小误差，红=大误差）")
    else:
        transformed_pcd.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([source_pcd, target_pcd, transformed_pcd])


# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载点云
    source_pcd = o3d.io.read_point_cloud("./my_data/27_cad_2.ply")
    target_pcd = o3d.io.read_point_cloud("./my_data/scan.ply")

    # 从 txt 文件读取转换矩阵
    transform_matrix = load_transform_matrix("transform_matrix.txt")

    # 应用变换
    transformed_pcd = apply_transform_to_pcd(source_pcd, transform_matrix)

    # 计算误差
    mean_err, rmse, max_err, distances = evaluate_registration_error(transformed_pcd, target_pcd)

    # 可视化
    visualize_registration(source_pcd, target_pcd, transformed_pcd, distances)

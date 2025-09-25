import numpy as np
from match import load_and_visualize  # 确保你保存的文件名是 match.py / alignment_utils.py


def transform_points(points_cad, T_cad2sensor, T_sensor2base, T_trans):
    """
    将 CAD 点集批量转换到机械臂 base 下
    points_cad: (N,3) CAD坐标下的点集
    T_cad2sensor: (4,4) CAD -> sensor 的齐次变换矩阵
    T_sensor2base: (4,4) sensor -> base 的齐次变换矩阵
    T_trans: (4,4) 电机位移增量矩阵

    return: (N,3) base 坐标下的点集
    """
    N = points_cad.shape[0]
    points_cad_h = np.hstack([points_cad, np.ones((N, 1))])
    T_total = T_trans @ T_sensor2base @ T_cad2sensor
    points_base_h = points_cad_h @ T_total.T
    return points_base_h[:, :3]


def make_translation(dx, dy, dz):
    """生成平移矩阵"""
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T


def make_rotation_z(theta_deg):
    """生成绕Z轴旋转矩阵"""
    theta = np.radians(theta_deg)
    T = np.eye(4)
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    return T


if __name__ == "__main__":
    # ========== 1. 先可视化 & 获取数据 ==========
    res_path = "./results/REG-DCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/eval_results/model_epoch19_T5_cosine_tudl_000001_noiseTrue_v1.pth"  # 修改成你的文件路径
    data = load_and_visualize(res_path, idx=195, show_normals=False)

    src_first5 = data["src_first5"]
    R_pred, t_pred = data["R_pred"], data["t_pred"]

    print("源点云前5个点 (CAD系):\n", src_first5)

    # ========== 2. 构造 CAD→sensor 矩阵 ==========
    T_cad2sensor = np.eye(4)
    T_cad2sensor[:3, :3] = R_pred
    T_cad2sensor[:3, 3] = t_pred

    # ========== 3. 定义 sensor→base 和 电机位移 ==========
    T_sensor2base = np.array([
        [1, 0, 0, 0.2],   # 平移0.2m
        [0, 1, 0, 0.1],   # 平移0.1m
        [0, 0, 1, 0.3],   # 平移0.3m
        [0, 0, 0, 1]
    ])

    T_trans = make_translation(0, 0, 0.05) @ make_rotation_z(5)  # 示例：推进 0.05m + Z 旋转5°

    # ========== 4. 转换前五个点到 base 系 ==========
    points_base = transform_points(src_first5, T_cad2sensor, T_sensor2base, T_trans)

    print("Base 坐标系下的前5个点:\n", points_base)

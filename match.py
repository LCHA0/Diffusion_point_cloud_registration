import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from utils.commons import load_data


def rotation_error(R1, R2):
    """计算旋转误差（角度，单位：度）"""
    R = R1 @ R2.T
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def plot_error_distribution(res):
    """绘制旋转 & 平移误差直方图"""
    Rs_gt, Rs_pred = np.array(res["Rs_gt"]), np.array(res["Rs_pred"])
    ts_gt, ts_pred = np.array(res["ts_gt"]), np.array(res["ts_pred"])

    rot_errs = [rotation_error(r_gt, r_pr) for r_gt, r_pr in zip(Rs_gt, Rs_pred)]
    trans_errs = np.linalg.norm(ts_gt - ts_pred, axis=1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(rot_errs, bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("Rotation error (deg)")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(trans_errs * 100, bins=50, color="salmon", edgecolor="black")
    plt.xlabel("Translation error (cm)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def visualize_alignment(res, idx=0, show_normals=False):
    """
    用 Open3D 可视化点云对齐效果，并返回前5个点 + 配准矩阵
    """
    if "src_pcd" not in res or "model_pcd" not in res:
        raise ValueError("结果文件中没有点云，请确认 test.py 已保存 src_pcd / model_pcd")

    src = np.array(res["src_pcd"][idx])
    model = np.array(res["model_pcd"][idx])

    R_pred = np.array(res["Rs_pred"][idx])
    t_pred = np.array(res["ts_pred"][idx])
    R_gt = np.array(res["Rs_gt"][idx])
    t_gt = np.array(res["ts_gt"][idx])

    print(f"\n=== 样本 idx={idx} 的姿态结果 ===")
    print("预测 Rs_pred:\n", R_pred)
    print("预测 ts_pred:\n", t_pred)
    print("GT Rs_gt:\n", R_gt)
    print("GT ts_gt:\n", t_gt)

    # === 计算旋转 & 平移误差 ===
    R_delta = R_gt @ R_pred.T
    cos_theta = (np.trace(R_delta) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cos_theta))
    trans_err = np.linalg.norm(t_gt - t_pred)

    print(f"旋转误差: {rot_err_deg:.3f} 度")
    print(f"平移误差: {trans_err*100:.2f} 厘米\n")  # 乘100假设输入是米

    # 应用预测 & GT 变换
    src_pred = src @ R_pred.T + t_pred
    src_gt = src @ R_gt.T + t_gt

    # 构造点云对象
    src_pred_pcd = o3d.geometry.PointCloud()
    src_pred_pcd.points = o3d.utility.Vector3dVector(src_pred)
    src_pred_pcd.paint_uniform_color([1, 0, 0])  # 红色 = 预测

    src_gt_pcd = o3d.geometry.PointCloud()
    src_gt_pcd.points = o3d.utility.Vector3dVector(src_gt)
    src_gt_pcd.paint_uniform_color([0, 0, 1])  # 蓝色 = GT

    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(model)
    model_pcd.paint_uniform_color([0, 1, 0])   # 绿色 = 模型

    geometries = [src_pred_pcd, src_gt_pcd]

    # 如果需要显示法向量
    if show_normals and "src_pcd_normal" in res and "model_pcd_normal" in res:
        src_normals = np.array(res["src_pcd_normal"][idx])
        model_normals = np.array(res["model_pcd_normal"][idx])
        src_pred_pcd.normals = o3d.utility.Vector3dVector(src_normals)
        model_pcd.normals = o3d.utility.Vector3dVector(model_normals)
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    o3d.visualization.draw_geometries(geometries)

    # === 返回数据 ===
    return {
        "src_first5": src[:5],
        "model_first5": model[:5],
        "R_pred": R_pred,
        "t_pred": t_pred,
        "R_gt": R_gt,
        "t_gt": t_gt
    }


def load_and_visualize(res_path, idx=0, show_normals=False):
    """
    外部可直接调用的主入口：
    1. 加载结果文件
    2. 打印样本总数
    3. 可视化 + 返回前5个点 + 配准矩阵
    """
    res = load_data(res_path)
    print("样本总数:", len(res["src_pcd"]))
    print(f"Loaded results from {res_path}")

    return visualize_alignment(res, idx=idx, show_normals=show_normals)

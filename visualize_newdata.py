import argparse
import numpy as np
import open3d as o3d
from utils.commons import load_data


def visualize_alignment(res, idx=0, show_normals=False):
    """只可视化预测结果 vs 模型点云"""
    if "src_pcd" not in res or "model_pcd" not in res:
        raise ValueError("结果文件中没有点云，请确认 test.py 已保存 src_pcd / model_pcd")

    src = np.array(res["src_pcd"][idx])       # 扫描点云
    model = np.array(res["model_pcd"][idx])   # CAD 点云
    R_pred = np.array(res["Rs_pred"][idx])    # 预测旋转
    t_pred = np.array(res["ts_pred"][idx])    # 预测平移

    print(f"\n=== 样本 idx={idx} 的预测姿态 ===")
    print("预测 Rs_pred:\n", R_pred)
    print("预测 ts_pred:\n", t_pred)

    # 应用预测变换到扫描点云
    src_pred = src @ R_pred.T + t_pred

    # 转换成 Open3D 点云
    src_pred_pcd = o3d.geometry.PointCloud()
    src_pred_pcd.points = o3d.utility.Vector3dVector(src_pred)
    src_pred_pcd.paint_uniform_color([1, 0, 0])  # 红色 = 扫描点云 (预测对齐)

    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(model)
    model_pcd.paint_uniform_color([0, 1, 0])     # 绿色 = CAD 模型点云

    geometries = [src_pred_pcd, model_pcd]

    # 如果需要显示法向量
    if show_normals and "src_pcd_normal" in res and "model_pcd_normal" in res:
        src_normals = np.array(res["src_pcd_normal"][idx])
        model_normals = np.array(res["model_pcd_normal"][idx])

        src_pred_pcd.normals = o3d.utility.Vector3dVector(src_normals)
        model_pcd.normals = o3d.utility.Vector3dVector(model_normals)

    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str, required=True, help="结果文件路径 .pth")
    parser.add_argument("--idx", type=int, default=0, help="展示的样本索引")
    parser.add_argument("--show_normals", action="store_true", help="是否显示法向量")
    args = parser.parse_args()

    res = load_data(args.res)
    print("样本总数:", len(res["src_pcd"]))
    print(f"Loaded results from {args.res}")

    visualize_alignment(res, idx=args.idx, show_normals=args.show_normals)

import open3d as o3d
import numpy as np
import copy


def load_and_preprocess(pcd_file, voxel_size=0.5):
    print(f"加载点云: {pcd_file}")
    pcd = o3d.io.read_point_cloud(pcd_file)
    print(f" 点数: {len(pcd.points)}")
    print(f" 范围: {np.asarray(pcd.get_min_bound())} ~ {np.asarray(pcd.get_max_bound())}")
    if voxel_size > 0:
        print(f" 下采样: {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    return pcd


def automatic_initial_alignment(source, target):
    """自动初始对齐（质心平移）"""
    print(" 自动初始对齐...")

    source_center = source.get_center()
    target_center = target.get_center()
    translation = target_center - source_center

    trans_init = np.identity(4)
    trans_init[0:3, 3] = translation

    print(f" 质心对齐平移: {translation}")
    return trans_init


def refine_registration(source, target, init_trans, voxel_size=1.0):
    print(" 执行精配准 (ICP)...")
    threshold = voxel_size * 1.5
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    print("✅ ICP 完成:")
    print(f"   fitness={reg_icp.fitness:.4f}, inlier_rmse={reg_icp.inlier_rmse:.4f}")
    return reg_icp.transformation


def save_transform_matrix(matrix, filename="transform_matrix.txt"):
    """以便于复制的格式保存变换矩阵"""
    with open(filename, "w") as f:
        f.write("[\n")
        for row in matrix:
            f.write("    [ " + ", ".join(f"{v: .18e}" for v in row) + "],\n")
        f.write("]\n")
    print(f" 已保存变换矩阵到: {filename}")


def main(cad_file, scan_file, voxel_size=0.5):
    print("=== CAD 与扫描点云配准（不裁剪） ===")

    cad_pcd = load_and_preprocess(cad_file, voxel_size)
    scan_pcd = load_and_preprocess(scan_file, voxel_size)

    # 自动初始对齐（质心平移）
    trans_init = automatic_initial_alignment(cad_pcd, scan_pcd)
    cad_transformed = copy.deepcopy(cad_pcd).transform(trans_init)

    print(" 可视化初始对齐结果")
    o3d.visualization.draw_geometries([
        cad_transformed.paint_uniform_color([1, 0, 0]),
        scan_pcd.paint_uniform_color([0, 1, 0])
    ])

    # 精配准（ICP）
    final_trans = refine_registration(cad_pcd, scan_pcd, trans_init, voxel_size)
    cad_final = copy.deepcopy(cad_pcd).transform(final_trans)

    print(" 可视化最终配准结果")
    o3d.visualization.draw_geometries([
        cad_final.paint_uniform_color([1, 0, 0]),
        scan_pcd.paint_uniform_color([0, 1, 0])
    ])

    # 保存结果
    o3d.io.write_point_cloud("aligned_cad.ply", cad_final)
    save_transform_matrix(final_trans, "transform_matrix.txt")
    print("✅ 已保存结果: aligned_cad.ply, transform_matrix.txt")


if __name__ == "__main__":
    cad_file = "./my_data/27_cad_2.ply"      # CAD 点云
    scan_file = "./my_data/scan.ply"      # 扫描点云
    main(cad_file, scan_file, voxel_size=0.5)

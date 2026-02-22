import open3d as o3d
import numpy as np
import copy
import argparse
import random
import json
import os
from datetime import datetime
from itertools import permutations


# ==========================================================
# 0️⃣ 固定随机种子（保证完全可复现）
# ==========================================================
np.random.seed(42)
random.seed(42)
o3d.utility.random.seed(42)


# ==========================================================
# 1️⃣ 基础预处理
# ==========================================================
def load_and_preprocess(pcd_file, voxel_size=0.5):
    print(f"加载点云: {pcd_file}")
    pcd = o3d.io.read_point_cloud(pcd_file)
    print(f" 点数: {len(pcd.points)}")
    if pcd.is_empty():
        raise RuntimeError(f"点云为空或不可读: {pcd_file}")

    if voxel_size > 0:
        print(f" 下采样: {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd


# ==========================================================
# 2️⃣ 特征计算
# ==========================================================
def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


# ==========================================================
# 3️⃣ 粗配准
# ==========================================================
def coarse_registration_deterministic(source, target, voxel_size=0.5):
    print("执行粗配准: FPFH + 高置信确定性匹配 + RANSAC")
    src_fpfh = compute_fpfh(source, voxel_size)
    tgt_fpfh = compute_fpfh(target, voxel_size)
    src_features = np.asarray(src_fpfh.data).T
    tgt_features = np.asarray(tgt_fpfh.data).T

    tgt_tree = o3d.geometry.KDTreeFlann(tgt_fpfh)
    src_tree = o3d.geometry.KDTreeFlann(src_fpfh)

    forward_matches = []
    for i, feat in enumerate(src_features):
        [_, idx, dist] = tgt_tree.search_knn_vector_xd(feat, 1)
        forward_matches.append((i, idx[0], dist[0]))

    correspondences = []
    for (i, j, d) in forward_matches:
        [_, idx_back, _] = src_tree.search_knn_vector_xd(tgt_features[j], 1)
        if idx_back[0] == i and d < 0.8:
            correspondences.append([i, j])
    corres = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, target, corres,
        max_correspondence_distance=voxel_size * 2.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)
    )
    print(f"RANSAC结果: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f}")
    return result.transformation


# ==========================================================
# 4️⃣ 精配准（Point-to-Plane ICP）
# ==========================================================
def refine_registration_icp_point2plane(source, target, init_trans, voxel_size=1.0, threshold_scale=1.5):
    threshold = voxel_size * threshold_scale
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )


# ==========================================================
# 5️⃣ 工具：平面基(u,v,n)
# ==========================================================
def build_plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = n.astype(np.float64)
    n /= (np.linalg.norm(n) + 1e-12)

    # 选一个不平行的参考向量
    if abs(n[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])

    u = np.cross(n, a)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-12)
    return u, v, n


def proj_to_plane_uv(points_xyz: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # (N,3) -> (N,2)
    return np.stack([points_xyz @ u, points_xyz @ v], axis=1)


def lift_uv_to_xyz(vec_uv: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # (2,) -> (3,)
    return vec_uv[0] * u + vec_uv[1] * v


# ==========================================================
# 6️⃣ 工具：提取凸起簇中心
# ==========================================================
def extract_convex_cluster_centers(
    pcd: o3d.geometry.PointCloud,
    plane_n: np.ndarray,
    plane_p0: np.ndarray,
    min_height: float,
    max_height: float,
    eps: float,
    min_points: int
):
    pts = np.asarray(pcd.points)
    dist = (pts - plane_p0) @ plane_n
    convex_idx = np.where((dist > min_height) & (dist < max_height))[0]
    if convex_idx.size < min_points:
        return []

    convex_pts = pts[convex_idx]
    convex_pcd = o3d.geometry.PointCloud()
    convex_pcd.points = o3d.utility.Vector3dVector(convex_pts)

    labels = np.array(convex_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    valid = labels >= 0
    if not np.any(valid):
        return []

    centers = []
    for lab in np.unique(labels[valid]):
        cluster = convex_pts[labels == lab]
        centers.append(cluster.mean(axis=0))
    return centers


# ==========================================================
# 7️⃣ 工具：2D刚体拟合（Kabsch）
# ==========================================================
def fit_rigid_2d(A: np.ndarray, B: np.ndarray):
    """
    A, B: (N,2)  A -> B
    return: R2(2,2), t2(2,)
    """
    assert A.shape == B.shape and A.shape[1] == 2
    N = A.shape[0]
    if N < 2:
        raise ValueError("2D刚体拟合至少需要2个点")

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cb - (R @ ca)
    return R, t


def best_permutation_rigid_2d(cad_uv: np.ndarray, scan_uv: np.ndarray, max_perm=40320):
    """
    枚举 scan 的对应（或 cad 的对应）寻找全局最优 2D 刚体变换
    N=4 时 24 种，非常轻。
    return: best_R2, best_t2, best_perm, best_rmse
    """
    N = cad_uv.shape[0]
    if N != scan_uv.shape[0]:
        raise ValueError("点数不一致，当前策略要求CAD簇数与scan簇数相同")

    perms = list(permutations(range(N)))
    if len(perms) > max_perm:
        # 极端情况下避免爆炸：只取一部分（这里不建议走到）
        perms = perms[:max_perm]

    best = None
    best_rmse = np.inf
    best_perm = None

    for perm in perms:
        B = scan_uv[list(perm), :]
        R2, t2 = fit_rigid_2d(cad_uv, B)
        pred = (cad_uv @ R2.T) + t2
        rmse = np.sqrt(np.mean(np.sum((pred - B) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best = (R2, t2)
            best_perm = perm

    return best[0], best[1], best_perm, best_rmse


# ==========================================================
# 8️⃣ 平面 + 凸起簇：平面内( Rz about n + Tx/Ty ) 微调
#     保留你的关键链路：ICP(1) -> 微调 -> ICP(2)
# ==========================================================
def micro_adjust_using_nozzle(
    scan_pcd: o3d.geometry.PointCloud,
    cad_final_after_icp1: o3d.geometry.PointCloud,
    voxel_size=0.5,
    plane_dist=None,
    min_height=2.0,
    max_height=20.0,
    eps=5.0,
    min_points=50
):
    print("执行方案一：RANSAC分割平面 + 凸起检测(全局簇匹配)进行平面内(Rz+Tx/Ty)微调...")

    if plane_dist is None:
        plane_dist = voxel_size

    # 1) scan 主平面
    plane_model, _ = scan_pcd.segment_plane(
        distance_threshold=plane_dist,
        ransac_n=3,
        num_iterations=2000
    )
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        print("  平面法向异常，跳过微调。")
        return np.eye(4)
    n /= n_norm
    p0 = -d * n
    print(f"  检测到平面法向: {n}, d={d:.3f}")

    # 2) 提取簇中心（scan & CAD）
    scan_centers = extract_convex_cluster_centers(
        scan_pcd, n, p0, min_height, max_height, eps, min_points
    )
    cad_centers = extract_convex_cluster_centers(
        cad_final_after_icp1, n, p0, min_height, max_height, eps, min_points
    )

    print(f"  scan簇数={len(scan_centers)}, CAD簇数={len(cad_centers)}")

    # 基本门槛：至少2个簇才可能约束平面内旋转
    if len(scan_centers) < 2 or len(cad_centers) < 2:
        print("  簇数不足(至少2)，跳过微调。")
        return np.eye(4)

    # 如果簇数不一致：先退回“单簇平移”（避免你当前这种旋转退化更严重）
    if len(scan_centers) != len(cad_centers):
        print("  簇数不一致，退回单簇平移策略（只修Tx/Ty，不引入Rz）...")
        u, v, _ = build_plane_basis(n)
        scan_uv = proj_to_plane_uv(np.asarray(scan_centers), u, v)
        cad_uv = proj_to_plane_uv(np.asarray(cad_centers), u, v)

        # 用最近的一对做平移
        best_t_uv = None
        best_err = np.inf
        best_pair = (-1, -1)
        for i in range(scan_uv.shape[0]):
            for j in range(cad_uv.shape[0]):
                t_uv = scan_uv[i] - cad_uv[j]
                err = np.linalg.norm(t_uv)
                if err < best_err:
                    best_err = err
                    best_t_uv = t_uv
                    best_pair = (i, j)

        t3 = lift_uv_to_xyz(best_t_uv, u, v)
        print(f"  → 单簇平移: scan#{best_pair[0]} ↔ cad#{best_pair[1]}, t={t3}, err={best_err:.2f}")

        T = np.eye(4)
        T[:3, 3] = t3
        return T

    # 3) 平面内2D坐标
    u, v, _ = build_plane_basis(n)
    scan_xyz = np.asarray(scan_centers, dtype=np.float64)
    cad_xyz = np.asarray(cad_centers, dtype=np.float64)
    scan_uv = proj_to_plane_uv(scan_xyz, u, v)
    cad_uv = proj_to_plane_uv(cad_xyz, u, v)

    # 4) 枚举对应 + 2D刚体拟合（锁住“围绕某个水嘴转圈”的自由度）
    R2, t2, perm, rmse = best_permutation_rigid_2d(cad_uv, scan_uv)
    print(f"  → 最优对应 perm={perm}, 2D拟合rmse={rmse:.3f}")

    # 5) 组装成3D变换：在(u,v)平面内旋转 + 平移，不改法向分量
    # R3 = [u v n] * [[R2,0],[0,1]] * [u v n]^T
    B = np.stack([u, v, n], axis=1)  # 3x3
    R_uv = np.eye(3)
    R_uv[:2, :2] = R2
    R3 = B @ R_uv @ B.T

    t3 = lift_uv_to_xyz(t2, u, v)

    T = np.eye(4)
    T[:3, :3] = R3
    T[:3, 3] = t3

    # 额外：确保不引入沿法向的平移（数值消抖）
    t3 = T[:3, 3]
    T[:3, 3] = t3 - (t3 @ n) * n

    print(f"  → 平面内微调：Rz_about_n + Tx/Ty, t={T[:3,3]}")
    return T


# ==========================================================
# 9️⃣ 保存矩阵
# ==========================================================
def save_transform_matrix(matrix, filename="transform_matrix.txt"):
    with open(filename, "w") as f:
        for row in matrix:
            f.write(" ".join(f"{v: .6f}" for v in row) + "\n")
    print(f"已保存变换矩阵到: {filename}")


def update_visualization_config(cad_aligned_path, scan_pcd_path, config_path="vis_config.json"):
    cfg = {
        "cad_pointcloud": os.path.abspath(cad_aligned_path),
        "scan_pointcloud": os.path.abspath(scan_pcd_path),
        "updated_at": datetime.now().isoformat()
    }
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"已更新可视化配置文件: {config_path}")


# ==========================================================
# 🔟 主流程
# ==========================================================
def main(cad_file, scan_file, voxel_size=0.5, enable_icp2=True):
    print("=== CAD 与扫描点云配准（确定性匹配 + 强制水嘴微调 + 再ICP） ===")
    cad_pcd = load_and_preprocess(cad_file, voxel_size)
    scan_pcd = load_and_preprocess(scan_file, voxel_size)

    # --- 粗配准 ---
    trans_init = coarse_registration_deterministic(cad_pcd, scan_pcd, voxel_size)
    cad_coarse = copy.deepcopy(cad_pcd).transform(trans_init)
    o3d.visualization.draw_geometries([
        cad_coarse.paint_uniform_color([1, 0, 0]),
        scan_pcd.paint_uniform_color([0, 1, 0])
    ], window_name="粗配准结果")

    # --- 第一次 ICP 精配准 ---
    print("\n--- 执行第一次 ICP 精配准 ---")
    reg_icp_1 = refine_registration_icp_point2plane(cad_pcd, scan_pcd, trans_init, voxel_size=voxel_size, threshold_scale=1.5)
    final_trans = reg_icp_1.transformation
    cad_final = copy.deepcopy(cad_pcd).transform(final_trans)
    print(f"ICP(1) 结果: fitness={reg_icp_1.fitness:.4f}, rmse={reg_icp_1.inlier_rmse:.4f}")

    o3d.visualization.draw_geometries([
        cad_final.paint_uniform_color([1, 0, 0]),
        scan_pcd.paint_uniform_color([0, 1, 0])
    ], window_name="ICP(1)结果")

    if enable_icp2:
        # --- 强制执行方案一微调：仍然发生在 ICP(1) 和 ICP(2) 之间（保留你的关键链路） ---
        print("\n强制执行方案一微调 (不论ICP质量) ...")
        T_corr = micro_adjust_using_nozzle(scan_pcd, cad_final, voxel_size=voxel_size)

        # 可视化用：对 cad_final 施加微调（与你原版一致）
        cad_final.transform(T_corr)
        # 合并矩阵：T_corr @ ICP1
        final_trans = T_corr @ final_trans

        o3d.visualization.draw_geometries([
            cad_final.paint_uniform_color([1, 0, 0]),
            scan_pcd.paint_uniform_color([0, 1, 0])
        ], window_name="水嘴微调后结果")

        # --- 第二次 ICP 精配准 (Refinement after 微调) ---
        print("\n--- 执行第二次 ICP 精配准 (Refinement after 微调) ---")
        # 如果你发现仍有“对应不够”，可以把 threshold_scale 临时调大到 3.0，再做一次小阈值 refine
        reg_icp_2 = refine_registration_icp_point2plane(cad_pcd, scan_pcd, final_trans, voxel_size=voxel_size, threshold_scale=1.5)
        final_trans = reg_icp_2.transformation
        cad_final = copy.deepcopy(cad_pcd).transform(final_trans)
        print(f"ICP(2) 结果: fitness={reg_icp_2.fitness:.4f}, rmse={reg_icp_2.inlier_rmse:.4f}")
    else:
        print("\n--- 跳过第二次 ICP 精配准（已禁用） ---")

    # --- 最终可视化 ---
    o3d.visualization.draw_geometries([
        cad_final.paint_uniform_color([1, 0, 0]),
        scan_pcd.paint_uniform_color([0, 1, 0])
    ], window_name="最终结果 (ICP2)")

    # --- 保存最终结果 ---
    aligned_cad_path = "aligned_cad.ply"
    transform_path = "transform_matrix.txt"

    o3d.io.write_point_cloud(aligned_cad_path, cad_final)
    save_transform_matrix(final_trans, transform_path)

    # 同步可视化配置：写对齐后的CAD文件路径
    update_visualization_config(
        cad_aligned_path=cad_file,
        scan_pcd_path=scan_file,
    )

    print("\n已保存最终结果并同步可视化配置")


# ==========================================================
# 11) 命令行入口
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAD 与扫描点云配准（确定性匹配 + 方案一微调）")
    parser.add_argument("--cad", type=str, default="./my_data/20260212data/cad.ply")
    parser.add_argument("--scan", type=str, default="./my_data/20260212data/scan.ply")
    parser.add_argument("--voxel", type=float, default=0.5)
    parser.add_argument(
        "--disable_icp2",
        action="store_true",
        help="禁用第二次 ICP 精配准（默认启用）"
    )

    args = parser.parse_args()
    main(args.cad, args.scan, voxel_size=args.voxel, enable_icp2=not args.disable_icp2)

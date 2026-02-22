import argparse
import numpy as np
import open3d as o3d
import os

# ============================================================
# IO
# ============================================================

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")
    return pcd

def load_transform(transform_path: str) -> np.ndarray:
    ext = os.path.splitext(transform_path)[1].lower()
    if ext == ".npy":
        T = np.load(transform_path)
    else:
        T = np.loadtxt(transform_path, delimiter="," if ext == ".csv" else None)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got {T.shape}")
    return T

def save_points_csv(points_xyz: np.ndarray, out_path: str):
    """
    Always write CSV with header: x,y,z
    Even if points_xyz is empty, still write header-only file.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    header = "x,y,z"
    if points_xyz.size == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
        return
    np.savetxt(out_path, points_xyz, delimiter=",", header=header, comments="", fmt="%.6f")

# ============================================================
# CAD pick & seeds
# ============================================================

def make_vis_pcd_for_picking(cad_full: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    cad_vis = cad_full.voxel_down_sample(voxel_size=voxel) if voxel > 0 else o3d.geometry.PointCloud(cad_full)
    cad_vis.paint_uniform_color([0.7, 0.7, 0.7])
    return cad_vis

def pick_one_point_get_z(pcd_vis: o3d.geometry.PointCloud) -> float:
    print("\n[Pick Mode]")
    print("  - Shift + Left Click to pick ONE point on CAD (downsampled for easier picking)")
    print("  - Press 'Q' (or ESC) to close window after picking\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick ONE point on CAD (Shift+Click), then press Q", width=1280, height=720)
    vis.add_geometry(pcd_vis)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if len(picked) < 1:
        raise RuntimeError("No point picked. Please pick one point with Shift+LeftClick.")
    idx = picked[0]
    pts = np.asarray(pcd_vis.points)
    z0 = float(pts[idx, 2])
    print(f"Picked point index={idx}, z0={z0:.6f}")
    return z0

def extract_plane_points_by_z(cad_full: o3d.geometry.PointCloud, z0: float, eps: float) -> o3d.geometry.PointCloud:
    pts = np.asarray(cad_full.points)
    mask = np.abs(pts[:, 2] - z0) <= eps
    indices = np.where(mask)[0]
    sel = cad_full.select_by_index(indices.tolist())
    print(f"Extracted {len(sel.points)} CAD points with |z - z0| <= {eps}")
    if sel.is_empty():
        raise RuntimeError("No CAD points found on the z-slice. Increase eps or pick another point.")
    return sel

def transform_points(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(pcd)
    out.transform(T)
    return out

def print_points_to_terminal(points_xyz: np.ndarray, print_all: bool, max_lines: int, title: str):
    n = points_xyz.shape[0]
    print(f"\n[{title}]")
    print(f"Total points: {n}")
    print("Format: x y z")
    if n == 0:
        return
    if print_all or n <= max_lines:
        for x, y, z in points_xyz:
            print(f"{x:.6f} {y:.6f} {z:.6f}")
    else:
        for i in range(max_lines):
            x, y, z = points_xyz[i]
            print(f"{x:.6f} {y:.6f} {z:.6f}")
        print(f"... ({n - max_lines} more lines not shown; use --print_all to print everything)")

# ============================================================
# ROI: lower hemisphere union around seeds
# ============================================================

def build_roi_in_scan_lower_hemisphere(
    scan_pcd: o3d.geometry.PointCloud,
    seed_pts_scan: np.ndarray,
    roi_r: float,
    down_axis: str = "z"
):
    if roi_r <= 0:
        raise ValueError("--roi_r must be > 0")
    if len(scan_pcd.points) == 0:
        raise RuntimeError("Scan point cloud has 0 points.")
    if seed_pts_scan.shape[0] == 0:
        raise RuntimeError("Seed points are empty.")
    if down_axis not in ("x", "y", "z"):
        raise ValueError("--down_axis must be one of: x, y, z")

    axis_id = {"x": 0, "y": 1, "z": 2}[down_axis]
    scan_pts = np.asarray(scan_pcd.points)

    kdtree = o3d.geometry.KDTreeFlann(scan_pcd)
    idx_set = set()

    for s in seed_pts_scan:
        _, idxs, _ = kdtree.search_radius_vector_3d(s, roi_r)
        if len(idxs) == 0:
            continue
        s_axis = s[axis_id]
        for ii in idxs:
            ii = int(ii)
            if scan_pts[ii, axis_id] <= s_axis:
                idx_set.add(ii)

    roi_indices = np.fromiter(idx_set, dtype=np.int64)
    roi_pcd = scan_pcd.select_by_index(roi_indices.tolist())
    return roi_pcd, roi_indices

# ============================================================
# Seeds grouping: DBSCAN
# ============================================================

def cluster_seeds_into_groups(seeds_in_scan: o3d.geometry.PointCloud, seed_eps: float, seed_min_points: int):
    if seeds_in_scan.is_empty():
        return [], np.array([], dtype=np.int64)

    labels = np.array(seeds_in_scan.cluster_dbscan(
        eps=seed_eps,
        min_points=seed_min_points,
        print_progress=False
    ))

    groups = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        if idx.size > 0:
            groups.append(idx)

    groups.sort(key=lambda x: x.size, reverse=True)

    print(f"\n[Seeds grouping] DBSCAN eps={seed_eps}, min_points={seed_min_points}")
    print(f"  Found groups: {len(groups)} (noise points: {(labels < 0).sum()})")
    for gi, gidx in enumerate(groups):
        print(f"  - group {gi}: {gidx.size} seed points")
    return groups, labels

def color_seeds_by_labels(seeds_in_scan: o3d.geometry.PointCloud, labels: np.ndarray) -> o3d.geometry.PointCloud:
    seeds_vis = o3d.geometry.PointCloud(seeds_in_scan)
    n = len(seeds_vis.points)
    colors = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        lab = int(labels[i])
        if lab < 0:
            colors[i] = np.array([0.5, 0.5, 0.5])  # noise gray
        else:
            r = ((lab * 37) % 255) / 255.0
            g = ((lab * 91) % 255) / 255.0
            b = ((lab * 151) % 255) / 255.0
            colors[i] = np.array([r, g, b])

    seeds_vis.colors = o3d.utility.Vector3dVector(colors)
    return seeds_vis

def visualize_all_groups_once(scan_pcd: o3d.geometry.PointCloud, seeds_in_scan: o3d.geometry.PointCloud, labels: np.ndarray):
    scan_vis = o3d.geometry.PointCloud(scan_pcd)
    scan_vis.paint_uniform_color([0.0, 1.0, 0.0])  # green
    seeds_vis = color_seeds_by_labels(seeds_in_scan, labels)

    o3d.visualization.draw_geometries(
        [scan_vis, seeds_vis],
        window_name="Group Overview: Scan(green) + Seeds(colored by group, noise=gray)",
        width=1280,
        height=720
    )

# ============================================================
# Plane fit from seeds (PCA) & projection
# ============================================================

def fit_plane_pca(points: np.ndarray):
    if points.shape[0] < 3:
        raise RuntimeError("Not enough points to fit plane.")

    p0 = points.mean(axis=0)
    X = points - p0
    C = (X.T @ X) / max(points.shape[0], 1)

    w, V = np.linalg.eigh(C)  # ascending
    n = V[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)

    u = V[:, 2]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    u = np.cross(v, n)
    u = u / (np.linalg.norm(u) + 1e-12)

    return p0, n, u, v

def point_plane_dist(points: np.ndarray, p0: np.ndarray, n: np.ndarray):
    return np.abs((points - p0) @ n)

def project_points_to_plane_2d(points: np.ndarray, p0: np.ndarray, u: np.ndarray, v: np.ndarray):
    X = points - p0
    return np.stack([X @ u, X @ v], axis=1)

def plane2d_to_3d(points2d: np.ndarray, p0: np.ndarray, u: np.ndarray, v: np.ndarray):
    return p0[None, :] + points2d[:, 0:1] * u[None, :] + points2d[:, 1:2] * v[None, :]

# ============================================================
# Curvature on scan (kNN PCA)
# ============================================================

def compute_curvature_knn(points_xyz: np.ndarray, kdtree: o3d.geometry.KDTreeFlann, query_pt: np.ndarray, k: int) -> float:
    _, idxs, _ = kdtree.search_knn_vector_3d(query_pt, k)
    if len(idxs) < 3:
        return 0.0
    neigh = points_xyz[np.asarray(idxs, dtype=np.int64)]
    mu = neigh.mean(axis=0)
    X = neigh - mu
    C = (X.T @ X) / max(len(neigh), 1)
    w = np.linalg.eigvalsh(C)  # ascending
    s = float(w[0] + w[1] + w[2])
    return float(w[0] / s) if s > 1e-12 else 0.0

def curvature_top_pct(scan_pcd: o3d.geometry.PointCloud, indices: np.ndarray, k: int, top_pct: float):
    if k < 10:
        raise ValueError("--curv_k must be >= 10 (recommend 30~120).")
    if not (0.0 < top_pct < 100.0):
        raise ValueError("--curv_pct must be in (0,100).")
    if len(indices) == 0:
        return np.array([], dtype=np.int64), 0.0

    scan_pts = np.asarray(scan_pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(scan_pcd)

    curv = np.zeros(len(indices), dtype=np.float64)
    for i, idx in enumerate(indices):
        p = scan_pts[int(idx)]
        curv[i] = compute_curvature_knn(scan_pts, kdtree, p, k)

    thr = float(np.percentile(curv, 100.0 - top_pct))
    sel = indices[curv >= thr]
    return sel, thr

# ============================================================
# RANSAC circle in 2D + coverage
# ============================================================

def circle_from_3pts(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return None

    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = float(np.linalg.norm(np.array([ux, uy]) - p1))
    return ux, uy, r

def ransac_circle_2d(points2d: np.ndarray, dist_thresh: float, iters: int, seed: int = 0):
    if points2d.shape[0] < 3:
        return None

    rng = np.random.default_rng(seed)
    N = points2d.shape[0]

    best_inliers = None
    best_num = -1
    best_model = None
    best_rmse = np.inf

    for _ in range(iters):
        ids = rng.choice(N, size=3, replace=False)
        model = circle_from_3pts(points2d[ids[0]], points2d[ids[1]], points2d[ids[2]])
        if model is None:
            continue
        cx, cy, r = model
        if not np.isfinite(r) or r <= 1e-6:
            continue

        d = np.linalg.norm(points2d - np.array([cx, cy]), axis=1)
        resid = np.abs(d - r)
        inliers = resid <= dist_thresh
        num = int(inliers.sum())
        if num < 10:
            continue

        rmse = float(np.sqrt(np.mean((resid[inliers]) ** 2)))
        if (num > best_num) or (num == best_num and rmse < best_rmse):
            best_num = num
            best_rmse = rmse
            best_inliers = inliers
            best_model = (cx, cy, r)

    if best_model is None:
        return None

    cx, cy, r = best_model
    return cx, cy, r, best_inliers, best_rmse

def angular_coverage(points2d: np.ndarray, cx: float, cy: float, bins: int = 72):
    if points2d.shape[0] == 0:
        return 0.0
    ang = np.arctan2(points2d[:, 1] - cy, points2d[:, 0] - cx)
    ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)
    bin_ids = np.floor(ang / (2.0 * np.pi) * bins).astype(int)
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    occ = np.unique(bin_ids).size
    return float(occ) / float(bins)

# ============================================================
# Horizontal plane (parallel to xoy): RANSAC on z = const
# ============================================================

def ransac_horizontal_plane_z(points_xyz: np.ndarray, dist_thresh: float, iters: int, seed: int = 0):
    if points_xyz.shape[0] == 0:
        return None, None, 0

    rng = np.random.default_rng(seed)
    z = points_xyz[:, 2]
    best_cnt = -1
    best_mask = None

    N = z.shape[0]
    for _ in range(max(1, iters)):
        zi = float(z[int(rng.integers(0, N))])
        mask = np.abs(z - zi) <= dist_thresh
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_mask = mask

    if best_mask is None or best_cnt <= 0:
        return None, None, 0

    z_plane = float(np.median(z[best_mask]))
    inlier_mask = np.abs(z - z_plane) <= dist_thresh
    return z_plane, inlier_mask, int(inlier_mask.sum())

def fit_global_big_plane_z(
    scan_pcd: o3d.geometry.PointCloud,
    dist_thresh: float,
    iters: int,
    voxel: float,
    min_inliers: int,
    min_ratio: float,
    seed: int = 0
) -> float:
    pcd_fit = scan_pcd.voxel_down_sample(voxel) if voxel and voxel > 0 else o3d.geometry.PointCloud(scan_pcd)
    pts = np.asarray(pcd_fit.points)
    if pts.shape[0] == 0:
        raise RuntimeError("Scan point cloud has 0 points (after downsample).")

    z_plane, _, inlier_cnt = ransac_horizontal_plane_z(
        pts, dist_thresh=dist_thresh, iters=iters, seed=seed
    )
    N = pts.shape[0]
    ratio = (inlier_cnt / max(1, N)) if z_plane is not None else 0.0

    print("\n[Global big-plane fit]")
    print(f"  Fit points used: {N} (pipe_fit_voxel={voxel})")
    print(f"  RANSAC result: z_plane={None if z_plane is None else f'{z_plane:.6f}'}  inliers={inlier_cnt}  ratio={ratio:.3f}  dist={dist_thresh}")

    ok = (z_plane is not None) and (inlier_cnt >= min_inliers) and (ratio >= min_ratio)
    if not ok:
        z_fallback = float(np.median(np.asarray(scan_pcd.points)[:, 2]))
        print(f"  WARNING: global plane not confident (need inliers>={min_inliers} and ratio>={min_ratio}).")
        print(f"  Fallback: z_plane = median(all scan z) = {z_fallback:.6f}")
        return z_fallback

    print("  Accepted global big-plane.")
    return float(z_plane)

# ============================================================
# Generate target circle points
# ============================================================

def generate_circle_2d(cx: float, cy: float, r: float, n_pts: int):
    if n_pts < 16:
        raise ValueError("--gen_n must be >= 16")
    theta = np.linspace(0.0, 2.0 * np.pi, num=n_pts, endpoint=False)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return np.stack([x, y], axis=1)

# ============================================================
# Visualizations
# ============================================================

def visualize_all_once(scan_pcd, roi_indices_all, ring_indices_all, target_xyz_all, title: str):
    roi_set = set(map(int, roi_indices_all.tolist())) if len(roi_indices_all) > 0 else set()
    ring_set = set(map(int, ring_indices_all.tolist())) if len(ring_indices_all) > 0 else set()
    roi_wo_ring = np.array(list(roi_set - ring_set), dtype=np.int64)

    scan_wo_roi = scan_pcd.select_by_index(list(roi_set), invert=True) if len(roi_set) > 0 else o3d.geometry.PointCloud(scan_pcd)
    roi_wo_ring_pcd = scan_pcd.select_by_index(roi_wo_ring.tolist()) if len(roi_wo_ring) > 0 else o3d.geometry.PointCloud()
    ring_pcd = scan_pcd.select_by_index(list(ring_set)) if len(ring_set) > 0 else o3d.geometry.PointCloud()

    scan_vis = o3d.geometry.PointCloud(scan_wo_roi)
    roi_vis = o3d.geometry.PointCloud(roi_wo_ring_pcd)
    ring_vis = o3d.geometry.PointCloud(ring_pcd)

    scan_vis.paint_uniform_color([0.0, 1.0, 0.0])   # green
    roi_vis.paint_uniform_color([1.0, 1.0, 0.0])    # yellow
    ring_vis.paint_uniform_color([0.0, 0.0, 1.0])   # blue

    geoms = [scan_vis, roi_vis, ring_vis]

    if target_xyz_all is not None and target_xyz_all.shape[0] > 0:
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz_all)
        target_pcd.paint_uniform_color([1.0, 0.0, 1.0])  # magenta
        geoms.append(target_pcd)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1280,
        height=720
    )

def visualize_final_scan_seeds_target(scan_pcd: o3d.geometry.PointCloud,
                                      seeds_in_scan: o3d.geometry.PointCloud,
                                      labels: np.ndarray,
                                      target_xyz_all: np.ndarray,
                                      title: str):
    """
    FINAL view: scan + seeds(colored) + generated target ring(magenta)
    """
    scan_vis = o3d.geometry.PointCloud(scan_pcd)
    scan_vis.paint_uniform_color([0.0, 1.0, 0.0])  # green

    seeds_vis = color_seeds_by_labels(seeds_in_scan, labels)

    geoms = [scan_vis, seeds_vis]

    if target_xyz_all is not None and target_xyz_all.shape[0] > 0:
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz_all)
        target_pcd.paint_uniform_color([1.0, 0.0, 1.0])  # magenta
        geoms.append(target_pcd)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1280,
        height=720
    )

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cad", required=True, help="CAD point cloud path")
    parser.add_argument("--scan", required=True, help="Scan point cloud path")
    parser.add_argument("--T", required=True, help="4x4 transform matrix file (.npy/.txt/.csv)")

    parser.add_argument("--voxel", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.5)

    parser.add_argument("--roi_r", type=float, required=True)
    parser.add_argument("--down_axis", type=str, default="z", choices=["x", "y", "z"])

    parser.add_argument("--no_pick", action="store_true")
    parser.add_argument("--z0", type=float, default=None)

    parser.add_argument("--print_all", action="store_true")
    parser.add_argument("--print_n", type=int, default=200)

    # Curvature (group-wise)
    parser.add_argument("--curv_k", type=int, default=50)
    parser.add_argument("--curv_pct", type=float, default=10.0)

    # Seed grouping
    parser.add_argument("--seed_eps", type=float, default=None)
    parser.add_argument("--seed_min_points", type=int, default=20)

    # plane & ring controls
    parser.add_argument("--plane_band", type=float, default=1.0)
    parser.add_argument("--radial_tol", type=float, default=0.6)

    # RANSAC circle
    parser.add_argument("--circle_iters", type=int, default=4000)
    parser.add_argument("--circle_seed", type=int, default=0)

    # Target circle
    parser.add_argument("--target_diam", type=float, required=True)
    parser.add_argument("--gen_n", type=int, default=360)

    # Global big-plane fit (horizontal z=const)
    parser.add_argument("--pipe_plane_dist", type=float, default=0.8)
    parser.add_argument("--pipe_plane_iters", type=int, default=3000)
    parser.add_argument("--pipe_plane_min_inliers", type=int, default=200)
    parser.add_argument("--pipe_plane_min_ratio", type=float, default=0.05)
    parser.add_argument("--pipe_fit_voxel", type=float, default=2.0)

    args = parser.parse_args()

    if args.target_diam <= 0:
        raise ValueError("--target_diam must be > 0")
    r_target = 0.5 * float(args.target_diam)

    cad_full = load_point_cloud(args.cad)
    scan_pcd = load_point_cloud(args.scan)
    T = load_transform(args.T)

    # ---- Global big-plane fit ONCE ----
    z_plane_global = fit_global_big_plane_z(
        scan_pcd=scan_pcd,
        dist_thresh=args.pipe_plane_dist,
        iters=args.pipe_plane_iters,
        voxel=args.pipe_fit_voxel,
        min_inliers=args.pipe_plane_min_inliers,
        min_ratio=args.pipe_plane_min_ratio,
        seed=args.circle_seed + 12345
    )
    print(f"[Global] Using z_plane_global = {z_plane_global:.6f} for ALL groups.\n")

    # pick z0
    if args.no_pick:
        if args.z0 is None:
            raise ValueError("When --no_pick is set, you must provide --z0")
        z0 = float(args.z0)
        print(f"Using provided z0={z0:.6f}")
    else:
        cad_vis = make_vis_pcd_for_picking(cad_full, voxel=args.voxel)
        print(f"CAD full points: {len(cad_full.points)} | CAD vis points: {len(cad_vis.points)}")
        z0 = pick_one_point_get_z(cad_vis)

    # seeds: CAD slice -> scan
    cad_sel_full = extract_plane_points_by_z(cad_full, z0=z0, eps=args.eps)
    seeds_in_scan = transform_points(cad_sel_full, T)
    seed_pts_all = np.asarray(seeds_in_scan.points)

    print_points_to_terminal(
        seed_pts_all,
        print_all=args.print_all,
        max_lines=max(0, args.print_n),
        title="Transformed seed points (CAD slice -> Scan)"
    )

    seed_eps = args.seed_eps if args.seed_eps is not None else max(3.0 * args.roi_r, 2.0 * args.voxel, 1e-6)

    groups, labels = cluster_seeds_into_groups(seeds_in_scan, seed_eps=seed_eps, seed_min_points=args.seed_min_points)
    if len(groups) == 0:
        raise RuntimeError("No seed groups found. Try adjusting --seed_eps and/or --seed_min_points.")

    # Overview
    visualize_all_groups_once(scan_pcd, seeds_in_scan, labels)

    scan_pts = np.asarray(scan_pcd.points)

    roi_all_set = set()
    ring_all_set = set()
    target_xyz_list = []

    for gi, seed_idx in enumerate(groups):
        seeds_group = seeds_in_scan.select_by_index(seed_idx.tolist())
        seeds_group_pts = np.asarray(seeds_group.points)

        print(f"\n==================== Group {gi} ====================")

        # 1) seed-plane
        p0, n, u, v = fit_plane_pca(seeds_group_pts)
        print(f"[Group {gi}] Seed-plane normal n = {n}  (unit)")

        # 2) ROI
        _, roi_indices_g = build_roi_in_scan_lower_hemisphere(
            scan_pcd, seeds_group_pts, args.roi_r, down_axis=args.down_axis
        )
        print(f"[Group {gi}] ROI points = {len(roi_indices_g)} (roi_r={args.roi_r}, lower hemi)")
        if len(roi_indices_g) == 0:
            continue

        for ii in roi_indices_g.tolist():
            roi_all_set.add(int(ii))

        roi_pts3d = scan_pts[roi_indices_g]

        # 3) plane-band in ROI
        dplane = point_plane_dist(roi_pts3d, p0, n)
        band_mask = dplane <= args.plane_band
        band_indices = roi_indices_g[band_mask]
        print(f"[Group {gi}] Plane-band points = {len(band_indices)} (plane_band={args.plane_band})")
        if len(band_indices) < 50:
            print(f"[Group {gi}] Too few plane-band points -> skip")
            continue

        # 4) curvature candidates
        curv_indices, curv_thr = curvature_top_pct(
            scan_pcd, band_indices, k=args.curv_k, top_pct=args.curv_pct
        )
        print(f"[Group {gi}] Curvature thr={curv_thr:.6e}, candidates={len(curv_indices)}")
        if len(curv_indices) < 20:
            print(f"[Group {gi}] Too few curvature candidates -> skip")
            continue

        # 5) circle in seed-plane (2D)
        cand_pts3d = scan_pts[curv_indices]
        cand_pts2d = project_points_to_plane_2d(cand_pts3d, p0, u, v)

        ransac = ransac_circle_2d(
            cand_pts2d,
            dist_thresh=args.radial_tol,
            iters=args.circle_iters,
            seed=args.circle_seed + gi
        )
        if ransac is None:
            print(f"[Group {gi}] Circle RANSAC failed -> skip")
            continue

        cx, cy, r_fit, _, rmse = ransac
        print(f"[Group {gi}] Circle: center2d=({cx:.3f},{cy:.3f}), r_fit={r_fit:.3f}, rmse={rmse:.3f}")
        print(f"[Group {gi}] Target: diam={args.target_diam:.3f} -> r_target={r_target:.3f}")

        # 6) detected ring (blue) with r_fit
        band_pts3d = scan_pts[band_indices]
        band_pts2d = project_points_to_plane_2d(band_pts3d, p0, u, v)
        d = np.linalg.norm(band_pts2d - np.array([cx, cy]), axis=1)
        resid = np.abs(d - r_fit)
        ring_mask = resid <= args.radial_tol
        ring_indices = band_indices[ring_mask]
        cov = angular_coverage(band_pts2d[ring_mask], cx, cy, bins=72) if ring_indices.size > 0 else 0.0
        print(f"[Group {gi}] Detected ring points={len(ring_indices)}, coverage≈{cov*100:.1f}%")

        for ii in ring_indices.tolist():
            ring_all_set.add(int(ii))

        # 7) generate target circle -> force onto GLOBAL big plane
        target2d = generate_circle_2d(cx, cy, r_target, n_pts=args.gen_n)
        target3d = plane2d_to_3d(target2d, p0, u, v)
        target3d[:, 2] = z_plane_global
        target_xyz_list.append(target3d)

    roi_indices_all = np.fromiter(roi_all_set, dtype=np.int64) if len(roi_all_set) > 0 else np.array([], dtype=np.int64)
    ring_indices_all = np.fromiter(ring_all_set, dtype=np.int64) if len(ring_all_set) > 0 else np.array([], dtype=np.int64)
    target_xyz_all = np.vstack(target_xyz_list) if len(target_xyz_list) > 0 else np.empty((0, 3), dtype=np.float64)

    print(f"\n[Global] ROI union points = {len(roi_indices_all)}")
    print(f"[Global] Detected ring union points = {len(ring_indices_all)}")
    print(f"[Global] Generated target points total = {target_xyz_all.shape[0]}")

    scan_base = os.path.splitext(os.path.basename(args.scan))[0]

    ring_xyz = scan_pts[ring_indices_all] if len(ring_indices_all) > 0 else np.empty((0, 3), dtype=np.float64)
    out_ring_csv = f"{scan_base}_ring_points.csv"
    save_points_csv(ring_xyz, out_ring_csv)
    print(f"[Global] Saved detected ring (blue) points: {out_ring_csv}  (rows={ring_xyz.shape[0]})")

    out_target_csv = f"{scan_base}_target_circle_points.csv"
    save_points_csv(target_xyz_all, out_target_csv)
    print(f"[Global] Saved generated target circle (magenta) points: {out_target_csv}  (rows={target_xyz_all.shape[0]})")

    # Existing global view
    visualize_all_once(
        scan_pcd,
        roi_indices_all,
        ring_indices_all,
        target_xyz_all,
        title=("ALL Groups: Scan(green,no ROI) + ROI(yellow) + DetectedRing(blue) + TargetCircle(magenta)\n"
               f"(band={args.plane_band}, ring_tol={args.radial_tol}, target_diam={args.target_diam}, z_plane={z_plane_global:.3f})")
    )

    # NEW final view: scan + seeds + target ring
    visualize_final_scan_seeds_target(
        scan_pcd=scan_pcd,
        seeds_in_scan=seeds_in_scan,
        labels=labels,
        target_xyz_all=target_xyz_all,
        title=("FINAL: Scan(green) + Seeds(colored) + TargetCircle(magenta)\n"
               f"(target_diam={args.target_diam}, z_plane={z_plane_global:.3f})")
    )

if __name__ == "__main__":
    main()

import numpy as np
import open3d as o3d

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TColgp import TColgp_Array1OfPnt


def random_points_in_triangle(p0, p1, p2, n):
    """在三角形内均匀随机采样 n 个点"""
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    mask = (u + v > 1)
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    return p0 + u * (p1 - p0) + v * (p2 - p0)


def step_to_ply_mesh_sampling(step_file, ply_file, density=200, max_points=1_000_000):
    # 1. 读取 STEP 文件
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != IFSelect_RetDone:
        raise RuntimeError("Error: cannot read STEP file")

    reader.TransferRoots()
    shape = reader.OneShape()

    # 2. 网格化 STEP 模型
    mesh = BRepMesh_IncrementalMesh(shape, 0.5)  # 容差控制精细度
    mesh.Perform()

    # 3. 遍历三角面
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    points = []

    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is None:
            exp.Next()
            continue

        triangles = triangulation.Triangles()

        for i in range(1, triangulation.NbTriangles() + 1):
            tri = triangles.Value(i)
            i1, i2, i3 = tri.Get()
            p1 = np.array(triangulation.Node(i1).Coord())
            p2 = np.array(triangulation.Node(i2).Coord())
            p3 = np.array(triangulation.Node(i3).Coord())

            # 面积
            area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            n_samples = max(int(area * density), 1)

            # 在三角形内采样
            tri_points = random_points_in_triangle(p1, p2, p3, n_samples)
            points.append(tri_points)

        exp.Next()

    points = np.vstack(points)
    print(f"生成点数: {len(points)}")

    # 4. Open3D 下采样（限制最大点数）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if len(points) > max_points:
        voxel_size = (pcd.get_max_bound() - pcd.get_min_bound()).max() / 200
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"⚠️ 超过 {max_points} 点，已下采样到 {len(pcd.points)}")

    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"✅ Saved {len(pcd.points)} points to {ply_file}")


if __name__ == "__main__":
    step_file = "20.stp"
    ply_file = "20.ply"
    step_to_ply_mesh_sampling(step_file, ply_file, density=200, max_points=1_000_000)

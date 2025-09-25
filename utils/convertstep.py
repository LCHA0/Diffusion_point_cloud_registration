import numpy as np
import open3d as o3d
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepTools import breptools_UVBounds

def step_to_ply_uniform(step_file, ply_file, u_samples=50, v_samples=50):
    # 1. 读取 STEP 文件
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != IFSelect_RetDone:
        raise RuntimeError("Error: cannot read STEP file")

    reader.TransferRoots()
    shape = reader.OneShape()

    # 2. 网格化 STEP 模型
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)  # 0.1 = 线性公差，越小越精细
    mesh.Perform()

    # 3. 遍历所有曲面，均匀采样 (u,v)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    points = []
    while exp.More():
        face = exp.Current()
        surf = BRepAdaptor_Surface(face)
        umin, umax, vmin, vmax = breptools_UVBounds(face)

        us = np.linspace(umin, umax, u_samples)
        vs = np.linspace(vmin, vmax, v_samples)

        for u in us:
            for v in vs:
                pnt = surf.Value(u, v)
                points.append([pnt.X(), pnt.Y(), pnt.Z()])

        exp.Next()

    points = np.array(points)

    # 4. 存储为 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_file, pcd)

    print(f"✅ Saved {len(points)} points to {ply_file}")


if __name__ == "__main__":
    step_file = "wob3000T.step"
    ply_file = "wob3000T.ply"
    step_to_ply_uniform(step_file, ply_file, u_samples=100, v_samples=100)

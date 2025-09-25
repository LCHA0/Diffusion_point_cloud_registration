import open3d as o3d
import numpy as np
import os
import glob

def mesh_to_pointcloud(mesh_file, output_file, num_points=10000, with_normals=True):
    """
    使用Open3D将mesh转换为点云
    
    参数:
    mesh_file: 输入mesh文件路径 (.stl, .obj, .ply等)
    output_file: 输出点云文件路径 (.ply, .pcd等) 
    num_points: 采样点数
    with_normals: 是否计算法向量
    """
    try:
        # 读取mesh文件
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        print(f"加载mesh: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角形")
        
        # 检查mesh是否为空
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh文件为空或格式不支持")
        
        # 表面均匀采样
        pointcloud = mesh.sample_points_uniformly(number_of_points=num_points)
        
        # 计算法向量(可选)
        if with_normals:
            pointcloud.estimate_normals()
        
        # 保存点云
        success = o3d.io.write_point_cloud(output_file, pointcloud)
        
        if success:
            print(f"✓ 转换成功!")
            print(f"  输出文件: {output_file}")
            print(f"  点云大小: {len(pointcloud.points)} 个点")
        else:
            print("✗ 保存失败，请检查输出路径和格式")
            
        return pointcloud
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return None

def visualize_pointcloud(pointcloud_file):
    """可视化点云"""
    try:
        pcd = o3d.io.read_point_cloud(pointcloud_file)
        print("正在启动3D可视化窗口...")
        o3d.visualization.draw_geometries([pcd], 
                                        window_name="点云可视化",
                                        width=800, 
                                        height=600)
    except Exception as e:
        print(f"可视化失败: {e}")

def advanced_sampling(mesh_file, num_points=10000):
    """展示不同的采样方法"""
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    # 方法1: 均匀采样
    pc_uniform = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # 方法2: 泊松圆盘采样(分布更均匀，但可能点数不精确)
    pc_poisson = mesh.sample_points_poisson_disk(
        number_of_points=num_points,
        init_factor=5
    )
    
    print(f"均匀采样: {len(pc_uniform.points)} 个点")
    print(f"泊松采样: {len(pc_poisson.points)} 个点")
    
    # 保存两种采样结果
    o3d.io.write_point_cloud("uniform_sampling.ply", pc_uniform)
    o3d.io.write_point_cloud("poisson_sampling.ply", pc_poisson)
    
    return pc_uniform, pc_poisson

def batch_convert_folder(input_folder, output_folder, num_points=10000):
    """批量转换文件夹中的mesh文件"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 支持的格式
    patterns = ["*.stl", "*.obj", "*.ply", "*.off"]
    
    converted_count = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(input_folder, pattern))
        
        for file_path in files:
            filename = os.path.basename(file_path)
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}.ply")
            
            print(f"\n处理: {filename}")
            result = mesh_to_pointcloud(file_path, output_path, num_points)
            
            if result is not None:
                converted_count += 1
    
    print(f"\n批量转换完成! 成功转换 {converted_count} 个文件")

def check_installation():
    """检查Open3D是否正确安装"""
    try:
        import open3d as o3d
        print(f"Open3D版本: {o3d.__version__}")
        print("✓ Open3D已正确安装")
        return True
    except ImportError:
        print("✗ Open3D未安装")
        print("请运行: pip install open3d")
        return False

def main():
    """主程序"""
    print("=" * 50)
    print("         Mesh转点云工具 (基于Open3D)")
    print("=" * 50)
    
    # 检查安装
    if not check_installation():
        return
    
    while True:
        print("\n请选择操作:")
        print("1. 转换单个文件")
        print("2. 批量转换文件夹")
        print("3. 高级采样对比")
        print("4. 可视化点云文件")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            # 单个文件转换
            print("\n--- 单个文件转换 ---")
            mesh_file = input("请输入mesh文件路径: ").strip().strip('"')
            
            if not os.path.exists(mesh_file):
                print("文件不存在!")
                continue
            
            # 自动生成输出文件名
            name, _ = os.path.splitext(mesh_file)
            output_file = f"{name}_pointcloud.ply"
            
            # 获取采样点数
            try:
                num_points = int(input("请输入采样点数 (默认10000): ") or "10000")
            except ValueError:
                num_points = 10000
            
            # 是否计算法向量
            with_normals = input("是否计算法向量? (y/n, 默认y): ").strip().lower() != 'n'
            
            # 执行转换
            pointcloud = mesh_to_pointcloud(mesh_file, output_file, num_points, with_normals)
            
            if pointcloud is not None:
                view = input("是否查看结果? (y/n): ").strip().lower()
                if view == 'y':
                    o3d.visualization.draw_geometries([pointcloud])
        
        elif choice == "2":
            # 批量转换
            print("\n--- 批量转换文件夹 ---")
            input_folder = input("请输入包含mesh文件的文件夹路径: ").strip().strip('"')
            
            if not os.path.exists(input_folder):
                print("文件夹不存在!")
                continue
            
            output_folder = input("请输入输出文件夹路径 (默认: pointclouds): ").strip() or "pointclouds"
            
            try:
                num_points = int(input("请输入采样点数 (默认10000): ") or "10000")
            except ValueError:
                num_points = 10000
            
            batch_convert_folder(input_folder, output_folder, num_points)
        
        elif choice == "3":
            # 高级采样对比
            print("\n--- 高级采样对比 ---")
            mesh_file = input("请输入mesh文件路径: ").strip().strip('"')
            
            if not os.path.exists(mesh_file):
                print("文件不存在!")
                continue
            
            try:
                num_points = int(input("请输入采样点数 (默认10000): ") or "10000")
            except ValueError:
                num_points = 10000
            
            pc_uniform, pc_poisson = advanced_sampling(mesh_file, num_points)
            
            view = input("是否对比查看两种采样结果? (y/n): ").strip().lower()
            if view == 'y':
                print("显示均匀采样结果...")
                o3d.visualization.draw_geometries([pc_uniform], window_name="均匀采样")
                print("显示泊松圆盘采样结果...")
                o3d.visualization.draw_geometries([pc_poisson], window_name="泊松采样")
        
        elif choice == "4":
            # 可视化点云
            print("\n--- 可视化点云文件 ---")
            pointcloud_file = input("请输入点云文件路径: ").strip().strip('"')
            
            if not os.path.exists(pointcloud_file):
                print("文件不存在!")
                continue
            
            visualize_pointcloud(pointcloud_file)
        
        elif choice == "5":
            print("再见!")
            break
        
        else:
            print("无效选项，请重新选择")

# 简单使用示例
def simple_example():
    """简单使用示例"""
    print("运行简单示例...")
    
    # 如果你有一个STL文件，可以直接运行这个例子
    input_file = "example.stl"  # 请替换为你的文件路径
    output_file = "output.ply"
    
    if os.path.exists(input_file):
        print(f"转换文件: {input_file}")
        pointcloud = mesh_to_pointcloud(input_file, output_file, num_points=5000)
        
        if pointcloud is not None:
            print("转换成功! 启动可视化...")
            o3d.visualization.draw_geometries([pointcloud])
    else:
        print(f"示例文件 {input_file} 不存在")
        print("请运行 main() 函数使用交互式界面")

if __name__ == "__main__":
    # 运行主程序
    # main()
    
    # 或者运行简单示例 (取消注释下面这行)
    simple_example()
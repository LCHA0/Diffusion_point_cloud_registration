import numpy as np

def parse_scanner_csv_to_ply(csv_file_path, output_ply_path, add_color=False):
    """
    解析激光扫描仪CSV数据并转换为PLY格式
    专门针对你提供的数据格式
    """
    
    # 尝试不同编码读取文件
    encodings = ['utf-8', 'gbk', 'utf-16', 'ascii', 'latin1']
    lines = None
    
    for encoding in encodings:
        try:
            with open(csv_file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                print(f"成功使用编码: {encoding}")
                break
        except:
            continue
    
    if lines is None:
        print("错误：无法读取文件")
        return
    
    # 找到包含 Y\X 的行
    header_line_idx = -1
    for i, line in enumerate(lines):
        if 'Y\\X' in line or 'Y/X' in line:
            header_line_idx = i
            break
    
    if header_line_idx == -1:
        print("错误：未找到Y\\X标题行")
        return
    
    print(f"找到数据标题在第{header_line_idx}行")
    
    # 解析X坐标（标题行）- 尝试不同分隔符
    header_line = lines[header_line_idx].strip()
    print(f"标题行内容: {repr(header_line)}")
    
    # 尝试不同的分隔符
    separators = ['\t', ',', ';', ' ', '  ', '   ']
    best_parts = []
    best_separator = None
    
    for sep in separators:
        parts = header_line.split(sep)
        if len(parts) > len(best_parts):
            best_parts = parts
            best_separator = sep
    
    print(f"使用分隔符: {repr(best_separator)}")
    print(f"标题行分割成{len(best_parts)}个部分")
    
    x_coords = []
    # 从第二个元素开始（跳过Y\X标记）
    for part in best_parts[1:]:
        part = part.strip()
        if part:  # 非空字符串
            try:
                x_val = float(part)
                x_coords.append(x_val)
            except ValueError:
                pass  # 忽略无法转换的值
    
    print(f"解析到{len(x_coords)}个X坐标")
    
    if len(x_coords) == 0:
        print("错误：没有找到有效的X坐标")
        print(f"标题行前10个部分: {best_parts[:10]}")
        return
        
    print(f"X坐标范围: {min(x_coords):.3f} 到 {max(x_coords):.3f}")
    
    # 解析数据行
    points = []
    processed_rows = 0
    
    for line_idx in range(header_line_idx + 1, len(lines)):
        line = lines[line_idx].strip()
        if not line:
            continue
            
        parts = line.split(best_separator)  # 使用检测到的最佳分隔符
        if len(parts) < 2:
            continue
        
        # 第一列是Y坐标
        y_str = parts[0].strip()
        if not y_str:
            continue
            
        try:
            y_coord = float(y_str)
        except ValueError:
            continue
        
        processed_rows += 1
        valid_points_in_row = 0
        
        # 解析这一行的Z值，使用相同的分隔符
        for col_idx in range(1, len(parts)):
            z_str = parts[col_idx].strip()
            if z_str:  # 非空字符串
                try:
                    z_val = float(z_str)
                    # 计算对应的X坐标索引
                    x_idx = col_idx - 1  # 减去Y列
                    if x_idx < len(x_coords):
                        x_coord = x_coords[x_idx]
                        points.append([x_coord, y_coord, z_val])
                        valid_points_in_row += 1
                except ValueError:
                    pass
        
        if processed_rows % 10 == 0:  # 每10行输出一次进度
            print(f"已处理{processed_rows}行数据，当前总点数: {len(points)}")
    
    print(f"总共处理了{processed_rows}行数据")
    print(f"总共解析到{len(points)}个有效数据点")
    
    if len(points) == 0:
        print("错误：没有找到任何有效的数据点")
        return
    
    points = np.array(points)
    
    print(f"\n点云统计信息:")
    print(f"  点数量: {len(points)}")
    print(f"  X范围: {points[:, 0].min():.3f} 到 {points[:, 0].max():.3f}")
    print(f"  Y范围: {points[:, 1].min():.3f} 到 {points[:, 1].max():.3f}")
    print(f"  Z范围: {points[:, 2].min():.3f} 到 {points[:, 2].max():.3f}")
    
    # 生成PLY文件
    if add_color:
        _write_colored_ply(points, output_ply_path)
    else:
        _write_basic_ply(points, output_ply_path)
    
    print(f"PLY文件已保存到: {output_ply_path}")

def _write_basic_ply(points, output_path):
    """写入基本PLY文件（无颜色）"""
    with open(output_path, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Generated from scanner CSV data\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")
        
        for point in points:
            ply_file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

def _write_colored_ply(points, output_path):
    """写入带颜色的PLY文件（根据高度着色）"""
    # 计算颜色（基于Z值的高度着色）
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    z_range = z_max - z_min
    
    print(f"高度着色: Z从{z_min:.3f}到{z_max:.3f}")
    
    with open(output_path, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Generated from scanner CSV data with height-based coloring\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        
        for point in points:
            # 计算归一化的高度值 (0-1)
            z_normalized = (point[2] - z_min) / z_range if z_range > 0 else 0
            
            # 使用彩虹色谱：蓝色(低) -> 绿色 -> 黄色 -> 红色(高)
            if z_normalized < 0.25:
                r = 0
                g = int(255 * z_normalized * 4)
                b = 255
            elif z_normalized < 0.5:
                r = 0
                g = 255
                b = int(255 * (1 - (z_normalized - 0.25) * 4))
            elif z_normalized < 0.75:
                r = int(255 * (z_normalized - 0.5) * 4)
                g = 255
                b = 0
            else:
                r = 255
                g = int(255 * (1 - (z_normalized - 0.75) * 4))
                b = 0
            
            # 确保颜色值在有效范围内
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            ply_file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")

def quick_test_parse(csv_file_path, max_lines=5):
    """
    快速测试解析功能，只处理前几行数据
    """
    print("=== 快速测试解析 ===")
    
    # 尝试不同编码
    encodings = ['utf-8', 'gbk', 'utf-16', 'ascii', 'latin1']
    lines = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            with open(csv_file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                used_encoding = encoding
                print(f"成功使用编码: {encoding}")
                break
        except:
            continue
    
    if lines is None:
        print("无法读取文件，尝试所有编码都失败")
        return
    
    # 找到Y\X行
    header_line_idx = -1
    for i, line in enumerate(lines):
        if 'Y\\X' in line or 'Y/X' in line:
            header_line_idx = i
            break
    
    if header_line_idx == -1:
        print("未找到Y\\X标题行")
        print("前30行内容:")
        for i, line in enumerate(lines[:30]):
            print(f"第{i}行: {repr(line.strip())}")
        return
    
    print(f"标题行在第{header_line_idx}行")
    header_line = lines[header_line_idx].strip()
    print(f"原始标题行: {repr(header_line)}")
    
    # 尝试不同的分隔符
    separators = ['\t', ',', ';', ' ', '  ', '   ']
    best_parts = []
    best_separator = None
    
    for sep in separators:
        parts = header_line.split(sep)
        if len(parts) > len(best_parts):
            best_parts = parts
            best_separator = sep
    
    print(f"最佳分隔符: {repr(best_separator)}")
    print(f"分割成{len(best_parts)}个部分")
    print(f"前10个部分: {best_parts[:10]}")
    
    # 解析X坐标
    x_coords = []
    for part in best_parts[1:]:  # 跳过第一个Y\X
        if part.strip():
            try:
                x_val = float(part.strip())
                x_coords.append(x_val)
            except:
                pass
    
    print(f"找到{len(x_coords)}个X坐标")
    print(f"前10个X坐标: {x_coords[:10]}")
    
    # 测试前几行数据
    test_points = []
    for i in range(1, min(max_lines + 1, len(lines) - header_line_idx)):
        line = lines[header_line_idx + i].strip()
        if not line:
            continue
        
        parts = line.split(best_separator)  # 使用检测到的分隔符
        if len(parts) < 2:
            continue
        
        try:
            y_coord = float(parts[0].strip())
            print(f"\n第{i}行: Y = {y_coord}")
            
            row_points = 0
            for j in range(1, len(parts)):
                if parts[j].strip():
                    try:
                        z_val = float(parts[j].strip())
                        if j-1 < len(x_coords):
                            x_coord = x_coords[j-1]
                            test_points.append([x_coord, y_coord, z_val])
                            row_points += 1
                    except:
                        pass
            
            print(f"  该行有{row_points}个有效点")
            
        except:
            print(f"第{i}行Y坐标解析失败")
    
    print(f"\n测试总结: 前{max_lines}行共找到{len(test_points)}个有效点")
    if test_points:
        test_points = np.array(test_points)
        print(f"X范围: {test_points[:, 0].min():.3f} - {test_points[:, 0].max():.3f}")
        print(f"Y范围: {test_points[:, 1].min():.3f} - {test_points[:, 1].max():.3f}")
        print(f"Z范围: {test_points[:, 2].min():.3f} - {test_points[:, 2].max():.3f}")
    
    return best_separator  # 返回检测到的分隔符

# 使用示例
if __name__ == "__main__":
    csv_file = "replay_19775_2025-9-18.csv"  # 替换为你的文件路径
    
    print("步骤1: 快速测试解析前5行")
    separator = quick_test_parse(csv_file, max_lines=5)
    
    if separator is None:
        print("无法确定分隔符，程序终止")
        exit()
    
    print("\n" + "="*60)
    print("步骤2: 完整转换（基本PLY）")
    parse_scanner_csv_to_ply(csv_file, "scanner_basic.ply", add_color=False)
    
    print("\n" + "="*60)
    print("步骤3: 完整转换（彩色PLY）")
    parse_scanner_csv_to_ply(csv_file, "scanner_colored.ply", add_color=True)
    
    print("\n转换完成！")
    print("输出文件:")
    print("- scanner_basic.ply: 基本点云")
    print("- scanner_colored.ply: 根据高度着色的点云")
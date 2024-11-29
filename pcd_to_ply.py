import os
import open3d as o3d

def convert_pcd_to_ply(input_pcd_file, output_ply_file):
    """
    將 .pcd 文件轉換為 .ply 文件。

    Args:
        input_pcd_file (str): 輸入的 .pcd 文件路徑。
        output_ply_file (str): 輸出的 .ply 文件路徑。
    """
    # 讀取 .pcd 文件
    try:
        pcd = o3d.io.read_point_cloud(input_pcd_file)
        print(f"成功讀取 PCD 文件：{input_pcd_file}")
    except Exception as e:
        print(f"讀取 PCD 文件失敗：{e}")
        return

    # 寫入 .ply 文件
    try:
        o3d.io.write_point_cloud(output_ply_file, pcd)
        print(f"成功將文件轉換並保存為 PLY 文件：{output_ply_file}")
    except Exception as e:
        print(f"寫入 PLY 文件失敗：{e}")

def convert_ply_to_ascii(input_file, output_file):
    """
    將二進制 PLY 文件轉換為 ASCII 格式。
    Args:
        input_file (str): 二進制 PLY 文件路徑。
        output_file (str): 輸出的 ASCII 格式 PLY 文件路徑。
    """
    # 讀取二進制 PLY 文件
    print(f"Reading binary PLY from {input_file}...")
    pcd = o3d.io.read_point_cloud(input_file)

    # 保存為 ASCII 格式
    print(f"Saving ASCII PLY to {output_file}...")
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print("Conversion complete!")

def process_ply_file(input_path, output_path):
    """
    處理 PLY 檔案，取整數座標並刪除重複的點，然後保存到新檔案。
    
    Args:
        input_path (str): 輸入 PLY 檔案路徑
        output_path (str): 輸出 PLY 檔案路徑
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # 找到 header 結束位置
    header_end_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end_idx = i
            break

    header = lines[:header_end_idx + 1]  # 保留 header
    data_lines = lines[header_end_idx + 1:]  # 剩下的是數據
    
    # 解析數據，取整數並刪除重複的點
    points = set()
    for line in data_lines:
        coords = tuple(map(int, map(float, line.strip().split())))
        points.add(coords)

    # 將處理後的數據保存到新檔案
    with open(output_path, 'w') as f:
        # 寫入 header
        f.writelines(header)
        # 更新 vertex 數量
        f.write(f"element vertex {len(points)}\n")
        for prop in header:
            if prop.startswith("property"):
                f.write(prop)
        f.write("end_header\n")
        # 寫入數據
        for point in sorted(points):
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"Processed and saved: {output_path}")

def process_all_ply_files(root_dir):
    """
    遍歷資料夾下的所有 PLY 檔案，處理符合條件的檔案。
    
    Args:
        root_dir (str): 資料夾路徑
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("pointcloud_") and file.endswith(".ply"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, file.replace("pointcloud_", "pointcloud_int_"))
                process_ply_file(input_path, output_path)
            elif file == "Town01_ascii.ply":
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, "Town01_ascii_int.ply")
                process_ply_file(input_path, output_path)

# 使用範例
data_dir = "data/"
input_pcd = "/home/perry/project/HDMaps/Town01.pcd"
output_ply = "output/Town01.ply"
input_ply = "output/Town01.ply"
output_ascii_ply = "output/Town01_ascii.ply"
# convert_pcd_to_ply(input_pcd, output_ply)
# convert_ply_to_ascii(input_ply, output_ascii_ply)
# process_all_ply_files(data_dir)

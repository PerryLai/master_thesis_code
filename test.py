import cv2
import os
import numpy as np
import open3d as o3d
import re
from scipy.spatial import KDTree
import plotly.graph_objects as go
def compute_camera_pose(feature_points_3d, feature_points_2d, camera_matrix):
    """
    使用 EPnP 計算相機的 6DoF 姿態。
    """
    dist_coeffs = np.zeros((4, 1))  # 無畸變
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=feature_points_3d,
        imagePoints=feature_points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        raise ValueError("EPnP 求解失敗")
    return rvec, tvec

def compute_camera_position_and_direction(rvec, tvec):
    """
    根據旋轉向量和平移向量計算相機的世界坐標位置和方向向量，並轉換方向角為方向向量。

    Args:
        rvec (numpy.ndarray): 旋轉向量。
        tvec (numpy.ndarray): 平移向量。

    Returns:
        camera_position (numpy.ndarray): 相機的世界坐標位置。
        camera_direction (numpy.ndarray): 相機的方向向量。
        rotation_angles (tuple): Pitch, Yaw, Roll 角度。
    """
    try:
        # 計算旋轉矩陣
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Carla -> OpenCV 坐標系轉換矩陣
        carla_to_opencv_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        rotation_matrix = np.dot(carla_to_opencv_matrix, rotation_matrix)

        # 檢查 rotation_matrix 是否有效
        if not np.isfinite(rotation_matrix).all():
            raise ValueError("計算的 rotation_matrix 包含無效值 (NaN 或 Inf)")

        # 計算相機的世界坐標位置
        camera_position = -np.dot(rotation_matrix.T, tvec).ravel()

        # 檢查 camera_position 是否有效
        if not np.isfinite(camera_position).all():
            raise ValueError("計算的 camera_position 包含無效值 (NaN 或 Inf)")

        # 計算相機的方向向量（光軸方向）
        camera_direction = rotation_matrix[:, 2]

        # 從旋轉矩陣計算 Pitch, Yaw, Roll
        pitch = np.arcsin(-rotation_matrix[2, 0]) * (180 / np.pi)
        yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * (180 / np.pi)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * (180 / np.pi)

        # 將方向角轉換為方向向量
        computed_direction = rotation_to_direction_vector(pitch, yaw, roll)

        return camera_position, computed_direction, (pitch, yaw, roll)

    except Exception as e:
        print(f"計算相機位置和方向時發生錯誤: {e}")
        return np.array([0, 0, 0]), np.array([0, 0, 1]), (0, 0, 0)

def parse_info_file(info_file):
    """
    解析 info_int.txt 文件，提取每個時間的相機位置和方向。

    Args:
        info_file (str): info_int.txt 文件的路徑。

    Returns:
        info_data (dict): 包含時間對應位置和方向的字典，例如:
                          {0: {"position": np.array([-52, 42, 0]), "direction": np.array([0, 89, 0])}, ...}
    """
    info_data = {}
    try:
        with open(info_file, 'r') as f:
            for line in f:
                # 使用正則表達式提取數據
                match = re.match(r"時間 (\d+): 位置: \(([-\d, ]+)\), 速度: \(([-\d, ]+)\), 旋轉: \(([-\d, ]+)\)", line)
                if match:
                    t = int(match.group(1))  # 提取時間
                    position = np.array(list(map(int, match.group(2).split(", "))))  # 提取位置
                    direction = np.array(list(map(int, match.group(4).split(", "))))  # 提取旋轉角度

                    # print(f"position: {position}")
                    # print(f"direction: {direction}")

                    info_data[t] = {"position": position, "direction": direction}
                else:
                    print(f"未匹配行: {line.strip()}")
    except FileNotFoundError:
        print(f"{info_file} 不存在，無法解析。")
    except Exception as e:
        print(f"解析 {info_file} 時發生錯誤: {e}")
    
    return info_data

def rotation_to_direction_vector(pitch, yaw, roll):
    """
    將 Pitch, Yaw, Roll 換算成世界坐標系中的方向向量。

    Args:
        pitch (float): 俯仰角（以度為單位）。
        yaw (float): 偏航角（以度為單位）。
        roll (float): 翻滾角（以度為單位）。

    Returns:
        np.ndarray: 世界坐標系中的方向向量。
    """
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    R = R_yaw @ R_pitch @ R_roll
    initial_vector = np.array([0, 0, -1])  # Carla 的相機前向
    direction_vector = R @ initial_vector
    return direction_vector

def process_data(data_dir, camera_matrix):
    """
    處理資料文件夾中的深度圖、RGB 圖和點雲文件，計算相機姿態。
    """
    for i in range(1):  # 遍歷 data/0 和 data/1
        vehicle_dir = os.path.join(data_dir, str(i), "vehicle", "vehicle_8")
        info_file = os.path.join(vehicle_dir, "info_int.txt")

        # 解析 info_int.txt
        info_data = parse_info_file(info_file)

        for j in range(10):  # 遍歷 depth_{j}, rgb_{j}, FP_{j}
            depth_file = f"{vehicle_dir}/depth_{j}.png"
            rgb_file = f"{vehicle_dir}/rgb_{j}.png"
            ply_file = f"{vehicle_dir}/pointcloud_{j}.ply"
            fp_file = f"{vehicle_dir}/FP_{j}.txt"

            print(f"\n正在處理 {fp_file}...")
            
            if not (os.path.exists(depth_file) and os.path.exists(rgb_file) and os.path.exists(ply_file) and os.path.exists(fp_file)):
                print(f"文件缺失，跳過: {depth_file}, {rgb_file}, {ply_file}, {fp_file}")
                continue

            depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

            feature_points_2d = []
            try:
                with open(fp_file, 'r') as f:
                    for line in f:
                        u, v, _ = map(float, line.split())
                        feature_points_2d.append([u, v])
            except Exception as e:
                print(f"讀取 {fp_file} 時發生錯誤: {e}")
                continue
            
            feature_points_2d = np.array(feature_points_2d, dtype=np.float32)
            print(f"2D 特徵點數量: {len(feature_points_2d)}")

            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            feature_points_camera = []
            valid_feature_points_2d = []
            for u, v in feature_points_2d:
                u, v = int(round(u)), int(round(v))
                if 0 <= u < depth_map.shape[1] and 0 <= v < depth_map.shape[0]:
                    z = depth_map[v, u]
                    if z > 0:
                        x_c = (u - cx) * z / fx
                        y_c = (v - cy) * z / fy
                        feature_points_camera.append([x_c, y_c, z])
                        valid_feature_points_2d.append([u, v])
            feature_points_camera = np.array(feature_points_camera, dtype=np.float32)
            valid_feature_points_2d = np.array(valid_feature_points_2d, dtype=np.float32)
            print(f"相機坐標系中的 3D 點數量: {len(feature_points_camera)}")

            pointcloud = o3d.io.read_point_cloud(ply_file)
            points_world = np.asarray(pointcloud.points)
            kdtree = KDTree(points_world)

            feature_points_world = []
            for point_camera in feature_points_camera:
                dist, index = kdtree.query(point_camera)
                feature_points_world.append(points_world[index])
            feature_points_world = np.array(feature_points_world, dtype=np.float32)

            if len(feature_points_world) < 4 or len(valid_feature_points_2d) < 4:
                print(f"{fp_file} 中有效點不足，無法求解相機姿態。")
                continue

            try:
                print("正在執行 solvePnP...")
                rvec, tvec = compute_camera_pose(feature_points_world, valid_feature_points_2d, camera_matrix)

                # 檢查 rvec 和 tvec 是否有效
                if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                    print(f"solvePnP 生成無效值 (NaN 或 Inf)，跳過該帧。")
                    continue

                camera_position, camera_direction, rotation_angles = compute_camera_position_and_direction(rvec, tvec)

                # 檢查計算結果是否在合理範圍內
                if np.any(np.abs(camera_position) > 1e6) or np.any(np.abs(camera_direction) > 1e3):
                    print(f"計算的相機位置或方向超出合理範圍，跳過該帧。")
                    continue

                # 將位置、方向和旋轉角度轉換為整數，避免溢位
                camera_position = np.clip(np.round(camera_position), -2**31, 2**31 - 1).astype(int)
                camera_direction = np.clip(np.round(camera_direction), -2**31, 2**31 - 1).astype(int)
                rotation_angles = np.clip(np.round(rotation_angles), -360, 360).astype(int)

                print("相機的世界坐標位置 (整數):", camera_position)
                print("相機的方向向量 (整數):", camera_direction)
                print("相機的旋轉角度 (Pitch, Yaw, Roll):", rotation_angles)

                if j in info_data:
                    recorded_position = info_data[j]["position"]
                    recorded_direction = info_data[j]["direction"]

                    diff_position = camera_position - recorded_position
                    diff_direction = camera_direction - recorded_direction  # 比較方向向量

                    print(f"info_int.txt 中的位置: {recorded_position}")
                    print(f"info_int.txt 中的方向向量: {recorded_direction}")
                    print(f"相機位置差值: {diff_position}")
                    print(f"相機方向向量差值: {diff_direction}")
                else:
                    print(f"無法找到 j = {j} 的對照數據。")


            except ValueError as e:
                print(f"處理 {fp_file} 時出錯: {e}")
            except Exception as e:
                print(f"未預期的錯誤發生: {e}")

def visualize_ply_file(input_ply_path, voxel_size=0.1):
    """
    從 PLY 檔案中讀取點雲並生成 3D 視覺化文件。
    支持對點雲進行下采樣以提高渲染性能。
    :param input_ply_path: PLY 檔案的完整路徑 (str)
    :param voxel_size: 下采樣體素大小 (float)，默認為 0.1。
    """
    # 確認檔案存在
    if not os.path.exists(input_ply_path):
        print(f"檔案不存在：{input_ply_path}")
        return
    
    # 讀取 PLY 檔案
    try:
        point_cloud = o3d.io.read_point_cloud(input_ply_path)
        print(f"成功讀取點雲，包含 {len(point_cloud.points)} 個點。")
    except Exception as e:
        print(f"讀取 PLY 檔案失敗：{e}")
        return

    # 檢查是否有有效點
    if len(point_cloud.points) == 0:
        print("點雲檔案中無有效點。")
        return

    # 點雲下采樣
    if voxel_size > 0:
        print(f"開始下采樣點雲，體素大小：{voxel_size}")
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        print(f"下采樣後的點雲包含 {len(point_cloud.points)} 個點。")

    # 獲取點雲座標
    points = np.asarray(point_cloud.points)

    # 獲取輸出檔名（與輸入同名但副檔名為 .html）
    output_file_path = input_ply_path.replace(".ply", ".html")

    # 使用 Plotly 生成 3D 視覺化
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], 
        y=points[:, 1], 
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # 根據 z 值上色
            colorscale='Viridis',  # 顏色方案
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D 點雲視覺化",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # 保存為 HTML 文件
    try:
        fig.write_html(output_file_path)
        print(f"3D 點雲視覺化已保存至：{output_file_path}")
    except Exception as e:
        print(f"保存視覺化檔案失敗：{e}")
        return

def visualize_ply_interactively(input_ply_path, voxel_size=0.1):
    """
    從 PLY 檔案中讀取點雲並使用交互式視窗顯示。
    :param input_ply_path: PLY 檔案的完整路徑 (str)
    :param voxel_size: 下采樣體素大小 (float)，默認為 0.1。
    """
    # 確認檔案存在
    if not os.path.exists(input_ply_path):
        print(f"檔案不存在：{input_ply_path}")
        return
    
    # 讀取 PLY 檔案
    try:
        point_cloud = o3d.io.read_point_cloud(input_ply_path)
        print(f"成功讀取點雲，包含 {len(point_cloud.points)} 個點。")
    except Exception as e:
        print(f"讀取 PLY 檔案失敗：{e}")
        return

    # 檢查是否有有效點
    if len(point_cloud.points) == 0:
        print("點雲檔案中無有效點。")
        return

    # 點雲下采樣
    if voxel_size > 0:
        print(f"開始下采樣點雲，體素大小：{voxel_size}")
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        print(f"下采樣後的點雲包含 {len(point_cloud.points)} 個點。")

    # 顯示點雲
    o3d.visualization.draw_geometries([point_cloud],
                                      window_name="3D 點雲視覺化",
                                      width=800, height=600)
if __name__ == "__main__":
    fx = 640
    fy = 640
    cx = 640
    cy = 360
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]])
    data_dir = "data"
    # input_ply = "output/Town01.ply"
    input_ply = "data/0/vehicle/vehicle_1/pointcloud_depth_4.ply"
    # process_data(data_dir, camera_matrix)
    # visualize_ply_file(input_ply)
    visualize_ply_interactively(input_ply)

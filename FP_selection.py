import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import numpy as np
import open3d as o3d

def scan_and_filter_dense_feature_points(data_dir, output_dir):
    """
    Scan the entire data/ directory for FP.txt files,
    extract all feature points, filter dense regions, and save the results.
    """
    feature_points = []

    # 遍歷 data/ 資料夾
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith("2D_FP_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")

                # 提取 FP.txt 中的特徵點
                with open(file_path, "r") as f:
                    for line in f:
                        coords = list(map(float, line.strip().split()))
                        feature_points.append(coords[:2])  # 提取 X, Y 平面座標

    # 若無特徵點則退出
    if not feature_points:
        print("No feature points found in the dataset.")
        return

    feature_points = np.array(feature_points)

    # 使用 DBSCAN 檢測密集區域
    clustering = DBSCAN(eps=5, min_samples=10).fit(feature_points)
    labels = clustering.labels_

    # 獲取所有聚類區域的點
    dense_feature_points = feature_points[labels != -1]  # 排除離群點
    print(f"Number of dense feature points: {len(dense_feature_points)}")

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 輸出密集特徵點到檔案
    output_fp_path = os.path.join(output_dir, "covisible_FP.txt")

    # 如果檔案已存在則刪除
    if os.path.exists(output_fp_path):
        print(f"Removing existing file: {output_fp_path}")
        os.remove(output_fp_path)

    with open(output_fp_path, "w") as f:
        for point in dense_feature_points:
            f.write(f"{point[0]} {point[1]}\n")
    print(f"Dense feature points saved at {output_fp_path}")

    # 繪製特徵點的散點圖
    plt.figure(figsize=(10, 10))
    plt.scatter(feature_points[:, 0], feature_points[:, 1], s=5, alpha=0.5, label="Feature Points")
    plt.scatter(dense_feature_points[:, 0], dense_feature_points[:, 1], s=20, color='red', alpha=0.8, label="Dense Points")

    # 為每個聚類區域繪製矩形框
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # 排除離群點
            continue
        cluster_points = feature_points[labels == label]
        x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
        y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()

        # 繪製矩形框以標註密集區域
        plt.gca().add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                edgecolor="blue",
                facecolor="none",
                linewidth=2,
                label="Dense Region" if label == 0 else None,
            )
        )

    plt.title("Feature Point Distribution and Dense Regions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # 確保輸出圖表目錄存在
    plot_output_path = os.path.join(output_dir, "feature_point_density.png")

    # 如果檔案已存在則刪除
    if os.path.exists(plot_output_path):
        print(f"Removing existing file: {plot_output_path}")
        os.remove(plot_output_path)

    # 儲存圖表到檔案
    plt.savefig(plot_output_path)
    plt.show()

    print(f"Feature point density plot saved at {plot_output_path}")

def extract_and_visualize_3d_feature_points(data_dir, output_dir):
    """
    提取 data/ 資料夾下所有 3D_FP_{i}.txt 文件中的特徵點，
    並在 3D 圖中標出特徵點分布密集的部分。

    Args:
        data_dir (str): 包含 3D_FP 文件的資料夾路徑。
        output_dir (str): 保存 3D 圖的目錄。
        
    Returns:
        feature_points (list of list): 所有特徵點的 (x, y, z) 坐標。
    """
    feature_points = []
    os.makedirs(output_dir, exist_ok=True)  # 創建輸出目錄

    # 遍歷 data/ 資料夾
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith("3D_FP_") and file.endswith(".txt"):  # 匹配文件格式
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")

                # 提取 3D_FP.txt 中的特徵點
                with open(file_path, "r") as f:
                    for line in f:
                        coords = list(map(float, line.strip().split()))
                        feature_points.append(coords)  # 提取 X, Y, Z 坐標

                # 可視化該文件的特徵點
                visualize_3d_feature_points(
                    np.array(feature_points), 
                    os.path.join(output_dir, f"{file.split('.')[0]}.png"),
                    title=f"Feature Points: {file}"
                )

    print(f"Total 3D feature points extracted: {len(feature_points)}")
    return feature_points

def visualize_3d_feature_points(points, save_path, title="3D Feature Points"):
    """
    在 3D 圖中可視化特徵點並標出分布密集區域。

    Args:
        points (numpy array): 特徵點的 3D 坐標 (N x 3)。
        save_path (str): 保存圖片的路徑。
        title (str): 圖表標題。
    """
    if len(points) == 0:
        print("No points to visualize.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 繪製所有特徵點
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1, label='Feature Points')

    # 標出密集區域
    dense_points = highlight_dense_regions(points)
    if len(dense_points) > 0:
        ax.scatter(
            dense_points[:, 0], dense_points[:, 1], dense_points[:, 2],
            c='r', s=5, label='Dense Regions'
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"3D visualization saved at {save_path}.")

def highlight_dense_regions(points, radius=0.5, density_threshold=10):
    """
    找出特徵點分布密集的部分。

    Args:
        points (numpy array): 特徵點的 3D 坐標 (N x 3)。
        radius (float): 鄰域半徑。
        density_threshold (int): 密集區域的鄰近點數量門檻。

    Returns:
        numpy array: 密集區域內的特徵點。
    """
    if len(points) == 0:
        return np.array([])

    tree = KDTree(points)
    dense_mask = np.array([
        len(tree.query_ball_point(point, radius)) >= density_threshold
        for point in points
    ])

    return points[dense_mask]

def filter_and_save_ply(data_dir, output_dir):
    """
    Filter the point cloud data in .ply files based on 3D feature points
    from 3D_FP_{i}.txt and save the filtered results into a new .ply file.

    Args:
        data_dir (str): Directory containing `pointcloud_{i}.ply` and `3D_FP_{i}.txt`.
        output_dir (str): Directory to save the filtered .ply file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filtered_points = []
    total_files = 0
    processed_files = 0

    # 計算總共的檔案數量
    for root, _, files in os.walk(data_dir):
        total_files += len([f for f in files if f.startswith("3D_FP_") and f.endswith(".txt")])

    print(f"Total 3D feature point files found: {total_files}")
    print("Starting the filtering process...")

    # 遍歷資料夾下的 pointcloud_{i}.ply 和 3D_FP_{i}.txt
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith("3D_FP_") and file.endswith(".txt"):
                processed_files += 1
                print(f"Processing file {processed_files}/{total_files}: {file}")

                # 對應的 3D feature points
                txt_path = os.path.join(root, file)
                timestamp = file.split("_")[2].split(".")[0]
                ply_file = os.path.join(root, f"pointcloud_{timestamp}.ply")
                
                if not os.path.exists(ply_file):
                    print(f"WARNING: Point cloud file {ply_file} not found, skipping.")
                    continue

                print(f"Reading feature points from {txt_path}...")
                # 讀取 3D_FP_{i}.txt
                feature_points = []
                with open(txt_path, "r") as f:
                    for line in f:
                        feature_points.append(list(map(float, line.strip().split())))

                feature_points = np.array(feature_points)

                print(f"Reading point cloud from {ply_file}...")
                # 讀取 pointcloud_{i}.ply
                pcd = o3d.io.read_point_cloud(ply_file)
                all_points = np.asarray(pcd.points)

                print(f"Filtering points for timestamp {timestamp}...")
                # 篩選出符合條件的點
                for feature_point in feature_points:
                    distances = np.linalg.norm(all_points - feature_point, axis=1)
                    nearest_idx = np.argmin(distances)
                    if distances[nearest_idx] < 0.01:  # 距離閾值
                        filtered_points.append(all_points[nearest_idx])

    if filtered_points:
        print("Saving filtered points to output file...")
        # 保存篩選後的點到新的 .ply 檔案
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
        filtered_ply_path = os.path.join(output_dir, "PCD_FP.ply")
        o3d.io.write_point_cloud(filtered_ply_path, filtered_pcd)
        print(f"Filtered point cloud saved to {filtered_ply_path}")
    else:
        print("No points were filtered. Output file not created.")

    # 可視化結果
    print("Visualizing the filtered point cloud...")
    o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")
    print("Process completed!")

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_and_save_top_1_percent_z(data_dir, output_file):
    """
    遍歷所有 data/ 資料夾中以 "3D_FP_" 開頭且 ".txt" 結尾的檔案，
    篩選 z 軸值在前 1% 的座標，保存結果到指定檔案，並生成 3D 可視化圖。

    Args:
        data_dir (str): 包含 3D 特徵點的資料夾。
        output_file (str): 保存篩選結果的檔案路徑。
    """
    all_points = []

    # 遍歷所有檔案
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith("3D_FP_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"讀取檔案: {file_path}")

                # 讀取檔案內容
                try:
                    points = np.loadtxt(file_path, dtype=int)  # 確保讀入為整數
                    if points.ndim == 1:  # 單行數據處理
                        points = points.reshape(1, -1)
                    if points.shape[1] == 3:  # 確保每行三個數值
                        all_points.append(points)
                    else:
                        print(f"跳過無效數據檔案: {file_path}")
                except Exception as e:
                    print(f"讀取檔案時出錯: {file_path}，錯誤: {e}")

    # 確保有有效數據
    if not all_points:
        print("沒有有效的 3D 點數據可以處理。")
        return

    # 合併所有點
    all_points = np.vstack(all_points)

    # 篩選 z 軸前 1% 的數據
    z_values = all_points[:, 2]
    z_threshold = np.percentile(z_values, 99)  # 前 1%
    top_1_percent_points = all_points[z_values >= z_threshold]

    print(f"篩選出 {len(top_1_percent_points)} 個 z 軸值在前 1% 的點。")

    # 保存篩選結果到檔案
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, top_1_percent_points, fmt="%d", delimiter=" ")
    print(f"篩選結果已保存至 {output_file}")

    # 3D 可視化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        top_1_percent_points[:, 0], 
        top_1_percent_points[:, 1], 
        top_1_percent_points[:, 2],
        c='r', marker='o', label="Top 1% z-axis"
    )
    ax.set_xlabel('X 軸')
    ax.set_ylabel('Y 軸')
    ax.set_zlabel('Z 軸')
    ax.legend()
    plt.title("3D 可視化圖 (z 軸前 1%)")
    plt.show()

# 主程式入口
if __name__ == "__main__":
    data_dir = "data"  # 輸入資料夾
    output_dir = "output"  # 圖表和篩選結果輸出資料夾
    target = "output/FP_town01.txt"
 
    # scan_and_filter_dense_feature_points(data_dir, output_dir)

    # filter_and_save_ply(data_dir, output_dir)

    # o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

    # all_3d_feature_points = extract_and_visualize_3d_feature_points(data_dir,output_dir)
    # print("Sample 3D feature points (x, y, z):", all_3d_feature_points[:5])
    visualize_and_save_top_1_percent_z(data_dir, target)
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np


def scan_and_filter_dense_feature_points(data_dir, output_dir):
    """
    Scan the entire data/ directory for FP.txt files,
    extract all feature points, filter dense regions, and save the results.
    """
    feature_points = []

    # 遍歷 data/ 資料夾
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith("FP.txt"):
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


# 主程式入口
if __name__ == "__main__":
    data_dir = "data"  # 輸入資料夾
    output_dir = "output"  # 圖表和篩選結果輸出資料夾
    scan_and_filter_dense_feature_points(data_dir, output_dir)




# import os
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# import numpy as np

# def scan_and_plot_feature_points(data_dir, output_dir):
#     """
#     Scan the entire data/ directory for FP.txt files,
#     extract all feature points, and plot their 2D distribution.
#     """
#     feature_points = []

#     # 遍歷 data/ 資料夾
#     for root, _, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith("FP.txt"):
#                 file_path = os.path.join(root, file)
#                 print(f"Processing {file_path}...")

#                 # 提取 FP.txt 中的特徵點
#                 with open(file_path, "r") as f:
#                     for line in f:
#                         coords = list(map(float, line.strip().split()))
#                         feature_points.append(coords[:2])  # 提取 X, Y 平面座標

#     # 若無特徵點則退出
#     if not feature_points:
#         print("No feature points found in the dataset.")
#         return

#     feature_points = np.array(feature_points)

#     # 繪製特徵點的散點圖
#     plt.figure(figsize=(10, 10))
#     plt.scatter(feature_points[:, 0], feature_points[:, 1], s=5, alpha=0.5, label="Feature Points")

#     # 使用 DBSCAN 檢測密集區域
#     clustering = DBSCAN(eps=5, min_samples=10).fit(feature_points)
#     labels = clustering.labels_

#     # 獲取所有聚類區域
#     unique_labels = set(labels)
#     for label in unique_labels:
#         if label == -1:  # -1 表示離群點
#             continue
#         cluster_points = feature_points[labels == label]
#         x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
#         y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()

#         # 繪製矩形框以標註密集區域
#         plt.gca().add_patch(
#             plt.Rectangle(
#                 (x_min, y_min),
#                 x_max - x_min,
#                 y_max - y_min,
#                 edgecolor="red",
#                 facecolor="none",
#                 linewidth=2,
#                 label="Dense Region" if label == 0 else None,
#             )
#         )

#     plt.title("Feature Point Distribution and Dense Regions")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.legend()
#     plt.grid(True)

#     # 確保輸出目錄存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 輸出圖表檔案名稱
#     output_path = os.path.join(output_dir, "feature_point_density.png")

#     # 如果檔案已存在則刪除
#     if os.path.exists(output_path):
#         print(f"Removing existing file: {output_path}")
#         os.remove(output_path)

#     # 儲存圖表到檔案
#     plt.savefig(output_path)
#     plt.show()

#     print(f"Feature point density plot saved at {output_path}")

# # 主程式入口
# if __name__ == "__main__":
#     data_dir = "data"  # 輸入資料夾
#     output_dir = "output"  # 圖表輸出資料夾
#     scan_and_plot_feature_points(data_dir, output_dir)

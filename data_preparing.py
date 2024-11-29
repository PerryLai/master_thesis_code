import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import numpy as np
import open3d as o3d

def reproject_3d_point_cloud(image_fp, intrinsic_matrix, extrinsic_params):
    """
    Reproject 2D image FP to 3D point cloud using EPnP results.
    """
    # Inverse projection: (X, Y, Z)_camera = K^-1 * (x, y, 1) * Z
    reprojection_3d = []
    for x, y, z in image_fp:
        pixel_coords = np.array([x, y, 1.0])
        camera_coords = np.linalg.inv(intrinsic_matrix).dot(pixel_coords) * z
        world_coords = np.dot(extrinsic_params[:3, :3], camera_coords) + extrinsic_params[:3, 3]
        reprojection_3d.append(world_coords)
    return np.array(reprojection_3d)

def perform_ndt_alignment(source_cloud, target_cloud, max_iterations=50):
    """
    Align source_cloud to target_cloud using NDT.
    """
    ndt = o3d.pipelines.registration.TransformationEstimationForPointToPoint()
    result = ndt.compute_transformation(source_cloud, target_cloud, max_iterations)
    print(f"NDT alignment complete with RMSE: {result.inlier_rmse}")
    return result.transformation

def calculate_vector_differences(reprojected_points, target_fp_points):
    """
    Calculate vector differences between reprojected points and target FP.
    """
    vectors = []
    for rp, tfp in zip(reprojected_points, target_fp_points):
        vectors.append(np.array(rp) - np.array(tfp))
    return np.array(vectors)

def detect_dynamic_objects(vectors, threshold=0.1):
    """
    Detect dynamic objects based on consistent motion vectors.
    """
    dynamic_objects = []
    for i in range(1, len(vectors)):
        diff = np.linalg.norm(vectors[i] - vectors[i - 1])
        if diff > threshold:
            dynamic_objects.append(i)
    return dynamic_objects

def calculate_accuracy(predicted_cloud, observed_cloud, threshold=0.05):
    """
    Calculate accuracy by comparing predicted and observed point clouds.
    """
    differences = []
    for pc, oc in zip(predicted_cloud, observed_cloud):
        diff = np.linalg.norm(pc - oc)
        differences.append(diff <= threshold)
    accuracy = np.mean(differences)
    print(f"Calculated accuracy: {accuracy * 100:.2f}%")
    return accuracy

def add_axes_to_visualization():
    """
    Create a coordinate frame to add axes for visualization.
    """
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0]
    )  # Adjust size as needed
    return axis

def extract_building_rooftop_points(input_ply):
    """
    從點雲中提取屬於建築物屋頂的點，保存結果，並用軸將它們視覺化。

    Args:
        input_ply (str): Path to the input PLY file.
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate output file paths
    ply_basename = os.path.splitext(os.path.basename(input_ply))[0]
    output_txt = os.path.join(output_dir, f"{ply_basename}_town01.txt")
    output_visualization = os.path.join(output_dir, f"{ply_basename}_rooftop_visualization.ply")

    print("Loading point cloud...")
    point_cloud = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(point_cloud.points)

    # Assume points with high Z values are rooftops
    threshold = np.percentile(points[:, 2], 99.2)  # Top 10% height
    rooftop_points = points[points[:, 2] >= threshold]

    print(f"Extracted {len(rooftop_points)} rooftop points.")
    np.savetxt(output_txt, rooftop_points, fmt='%f', header="X Y Z")
    print(f"Rooftop points saved to {output_txt}.")

    # Visualization
    rooftop_pcd = o3d.geometry.PointCloud()
    rooftop_pcd.points = o3d.utility.Vector3dVector(rooftop_points)
    o3d.io.write_point_cloud(output_visualization, rooftop_pcd)
    print(f"3D visualization saved to {output_visualization}.")

    # Read back from output_txt to ensure correct format and visualize
    loaded_points = np.loadtxt(output_txt, skiprows=1)  # Skip header
    loaded_pcd = o3d.geometry.PointCloud()
    loaded_pcd.points = o3d.utility.Vector3dVector(loaded_points)
    
    # Add axes to visualization
    print("Adding axes to visualization...")
    try:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        print("Axes successfully created.")
    except Exception as e:
        print(f"Failed to create axes: {e}")
        axes = None

    print(f"Visualizing points from {output_txt} with axes...")
    if axes:
        o3d.visualization.draw_geometries(
            [loaded_pcd, axes],
            window_name="Rooftop Points Visualization",
            width=800,
            height=600,
            point_show_normal=False
        )
    else:
        print("Rendering without axes due to an issue with coordinate frame creation.")
        o3d.visualization.draw_geometries(
            [loaded_pcd],
            window_name="Rooftop Points Visualization",
            width=800,
            height=600,
            point_show_normal=False
        )

def process_file_to_integers(input_file, output_file):
    """
    將文件中的所有數字取整後保存到新文件。
    
    Args:
        input_file (str): 輸入文件的路徑。
        output_file (str): 輸出文件的路徑。
    """
    input_file = "output/PCD_FP_town01.txt"  # 替換為你的輸入文件名稱
    output_file = 'output/PCD_FP_town01_int.txt'  # 替換為你的輸出文件名稱
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip():  # 確保非空行
                processed_line = ' '.join(
                    str(int(float(num))) if num.replace('.', '', 1).replace('-', '', 1).isdigit() else num
                    for num in line.split()
                )
                outfile.write(processed_line + '\n')

# 示例使用
input_file_path = 'input.txt'  # 替換為你的輸入文件名稱
output_file_path = 'output.txt'  # 替換為你的輸出文件名稱
process_file_to_integers(input_file_path, output_file_path)

import open3d as o3d
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([axis], window_name="Coordinate Axes")

if __name__ == "__main__":
    data_dir = "data"  # 輸入資料夾
    output_dir = "output"  # 圖表和篩選結果輸出資料夾
    filtered_pcd = "output/PCD_FP.ply"
    # extract_building_rooftop_points(filtered_pcd)
    input_file_path = "output/PCD_FP_town01.txt"  # 替換為你的輸入文件名稱
    output_file_path = 'output/PCD_FP_town01_int.txt'  # 替換為你的輸出文件名稱
    process_file_to_integers(input_file_path, output_file_path)

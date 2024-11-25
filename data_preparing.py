import open3d as o3d
import numpy as np
import cv2
import os

def process_pcd(pcd_file, fp_txt_folder):
    """
    1. Downsample PCD file to 1 point per cubic meter.
    2. Filter PCD to retain only overlapping points with FP.
    3. Use EPnP to estimate projection parameters.
    """

    # Step 1: Downsampling PCD
    print("Loading PCD file...")
    pcd = o3d.io.read_point_cloud(pcd_file)
    print(f"Original PCD size: {len(pcd.points)}")

    voxel_size = 1.0  # Define voxel size for downsampling
    print(f"Downsampling PCD with voxel size {voxel_size}...")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled PCD size: {len(downsampled_pcd.points)}")

    # Step 2: Retain overlapping points with FP
    print("Loading FP coordinates from txt files...")
    fp_coordinates = []
    for file in os.listdir(fp_txt_folder):
        if file.endswith("_FP.txt"):
            file_path = os.path.join(fp_txt_folder, file)
            with open(file_path, "r") as f:
                for line in f:
                    coords = np.array([float(c) for c in line.strip().split()])
                    fp_coordinates.append(coords)

    fp_coordinates = np.array(fp_coordinates)
    print(f"Loaded {len(fp_coordinates)} feature points.")

    # Convert PCD points to numpy array
    downsampled_pcd_points = np.asarray(downsampled_pcd.points)

    # Find overlapping points (within a small distance threshold)
    threshold = 0.5  # Distance threshold for matching
    print(f"Matching PCD points with FP within threshold {threshold}...")
    overlapping_points = []
    for fp in fp_coordinates:
        distances = np.linalg.norm(downsampled_pcd_points - fp, axis=1)
        if np.min(distances) < threshold:
            overlapping_points.append(fp)

    overlapping_points = np.array(overlapping_points)
    print(f"Number of overlapping points: {len(overlapping_points)}")

    # Step 3: EPnP calculation
    print("Evaluating EPnP projection...")
    # Simulate 2D pixel coordinates for overlapping points
    image_size = (1920, 1080)
    intrinsic_matrix = np.array([
        [1000, 0, image_size[0] / 2],
        [0, 1000, image_size[1] / 2],
        [0, 0, 1]
    ])
    extrinsic_matrix = np.eye(4)  # Assume identity for simplicity

    # Simulate pixel projections
    projected_points = []
    for point in overlapping_points:
        world_point = np.append(point, 1)  # Homogeneous coordinates
        camera_coords = np.dot(extrinsic_matrix, world_point)[:3]
        pixel_coords = np.dot(intrinsic_matrix, camera_coords)
        pixel_coords /= pixel_coords[2]  # Normalize by depth
        projected_points.append(pixel_coords[:2])

    projected_points = np.array(projected_points)
    image_points = projected_points[:, :2]  # 2D image points
    world_points = overlapping_points  # 3D world points

    # Solve EPnP
    print("Solving EPnP...")
    _, rvec, tvec = cv2.solvePnP(
        objectPoints=world_points,
        imagePoints=image_points,
        cameraMatrix=intrinsic_matrix,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    print("EPnP Results:")
    print("Rotation Matrix:")
    print(rotation_matrix)
    print("Translation Vector:")
    print(tvec)

    # Return key projection parameters
    return {
        "rotation_matrix": rotation_matrix,
        "translation_vector": tvec,
        "intrinsic_matrix": intrinsic_matrix
    }


# Example usage
if __name__ == "__main__":
    pcd_file = "data/map_image_Town01.pcd"  # Replace with your PCD file path
    fp_txt_folder = "output/covisible_FP.txt"  # Replace with your FP TXT folder path

    print("Processing PCD and FP data...")
    epnp_result = process_pcd(pcd_file, fp_txt_folder)
    print("EPnP computation completed.")

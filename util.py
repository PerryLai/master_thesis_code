import numpy as np
import carla
import cv2

def clear_existing_actors(world):
    """
    清除所有已存在的車輛和行人。
    """
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.*')
    controllers = actors.filter('controller.*')

    for vehicle in vehicles:
        vehicle.destroy()
    for walker in walkers:
        walker.destroy()
    for controller in controllers:
        controller.destroy()

    print(f"清理完成：共移除 {len(vehicles)} 輛車輛，{len(walkers)} 名行人，及 {len(controllers)} 個控制器。")

def save_pointcloud_to_ply(points, filename):
    """
    保存點雲到 .ply 文件。
    """
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"點雲已保存至 {filename}")

def attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth'):
    """
    將指定類型的相機附加到車輛上。
    """
    try:
        # print("camera_bp")
        camera_bp = world.get_blueprint_library().find(camera_type)
        # print("camera_bp.set_attribute")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")
        # print("transform")
        transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
        # 確保 transform 和 vehicle 合法
        assert camera_bp is not None, "Camera blueprint 不存在！"
        assert vehicle is not None, "Vehicle 不存在！"
        # print("camera")
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        # print("return camera")
        return camera
    except RuntimeError as e:
        print(f"將 {camera_type} 附加到車輛失敗：{e}")
        return None

def attach_lidar_to_vehicle(world, vehicle, lidar_range=50.0, channels=32, points_per_second=100000, rotation_frequency=10):
    """
    在指定車輛上附加 LiDAR 感測器。
    """
    print("Attach lidar start!")
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', str(lidar_range))
    lidar_bp.set_attribute('channels', str(channels))
    lidar_bp.set_attribute('points_per_second', str(points_per_second))
    lidar_bp.set_attribute('rotation_frequency', str(rotation_frequency))

    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))  # LiDAR 放在車頂
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    print("Attach lidar success!")
    return lidar

def get_camera_intrinsic(camera):
    """
    獲取相機的內參矩陣。
    """
    fov = float(camera.attributes['fov'])
    width = int(camera.attributes['image_size_x'])
    height = int(camera.attributes['image_size_y'])
    focal_length = width / (2.0 * np.tan(np.radians(fov / 2.0)))
    cx = width / 2.0
    cy = height / 2.0

    return np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

def detect_feature_points(image):
    """
    使用 ORB（Oriented FAST and Rotated BRIEF）檢測影像中的特徵點。

    Args:
        image: 用於特徵檢測的輸入影像。

    Returns:
        以 (x, y) 元組形式返回 2D 特徵點列表。
    """
    # print("--------檢測特徵點------------")
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(gray, None)
    return [kp.pt for kp in keypoints]

def estimate_camera_pose_2d_3d(image_points, world_points):
    """
    使用 2D-3D 對應關係估計相機的姿態（旋轉和平移向量）。

    Args:
        image_points (np.ndarray): 圖像中檢測到的 2D 特徵點。
        world_points (np.ndarray): 對應的 3D 世界座標點。

    Returns:
        rvec (np.ndarray): 旋轉向量。
        tvec (np.ndarray): 平移向量。
    """
    assert len(image_points) >= 4, "相機姿態估計需要至少 4 個圖像點。"
    assert len(world_points) >= 4, "相機姿態估計需要至少 4 個世界座標點。"
    
    # 轉換為所需的格式，並取整數位
    image_points = np.array([[int(x), int(y)] for x, y in image_points], dtype=np.float32)
    world_points = np.array([[int(x), int(y), int(z)] for x, y, z in world_points], dtype=np.float32)
    
    # 相機內參矩陣（根據設置進行修改）
    camera_matrix = np.array([
        [1000, 0, 640],  # fx, 0, cx
        [0, 1000, 360],  # 0, fy, cy
        [0, 0, 1]        # 0, 0, 1
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # 假設沒有鏡頭畸變

    # 使用 EPnP 演算法求解 PnP 問題
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=world_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    print(f"--------------------成功估計相機姿態: {success}---------------------")
    if not success:
        raise RuntimeError("---------------EPnP 未能成功估計相機姿態。----------------------")
    return rvec, tvec

def save_ply_file(depth_image, intrinsic, transform, ply_filename):
    """
    從深度影像生成 .ply 文件。

    Args:
        depth_image: CARLA 深度影像。
        intrinsic: 相機內參矩陣。
        transform: 相機外部變換矩陣。
        ply_filename: 保存 .ply 文件的路徑。
    """
    depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    depth_array = depth_array * 1000.0 / 255.0  # 轉換為公尺
    points = []

    height, width = depth_array.shape
    for y in range(height):
        for x in range(width):
            depth = depth_array[y, x]
            if depth <= 0:
                continue  # 跳過無效深度點
            pixel_coords = np.array([x, y, 1.0])
            camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
            world_coords = transform.transform(carla.Location(
                x=int(camera_coords[0]),  # 取整數位
                y=int(camera_coords[1]),  # 取整數位
                z=int(camera_coords[2])   # 取整數位
            ))
            points.append(f"{world_coords.x} {world_coords.y} {world_coords.z} 255 255 255\n")

    with open(ply_filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        f.writelines(points)

    print(f"點雲數據已保存為 .ply 文件：{ply_filename}")


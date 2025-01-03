import carla
import os
import time
import random
import cv2
import numpy as np
import tqdm

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

    print(f"Cleared {len(vehicles)} vehicles, {len(walkers)} walkers, and {len(controllers)} controllers.")

def generate_simulation_data(world, experiment_id, dirname, duration=3, interval=1):

    # 設置同步模擬模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0
    settings.max_substep_delta_time = 0.05
    settings.max_substeps = 20
    world.apply_settings(settings)

    # 嘗試多次應用設置
    for attempt in range(30):
        world.apply_settings(settings)
        updated_settings = world.get_settings()
        if updated_settings.synchronous_mode and updated_settings.fixed_delta_seconds == 1.0:
            print("Settings applied successfully.")
            break
        time.sleep(0.1)  # 短暫等待重試
    else:
        print("Failed to apply settings after 30 attempts.")

    # Generate vehicles and walkers, record their data, and capture images with feature points.
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    walker_bp_library = world.get_blueprint_library().filter('*walker*')
    spawn_points = world.get_map().get_spawn_points()

    # Create directories
    vehicle_dir = os.path.join(dirname, "vehicle")
    walker_dir = os.path.join(dirname, "walkers")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(walker_dir, exist_ok=True)

    destroyed_vehicles = set()
    destroyed_walkers = set()

    vehicles, walkers = [], []

    # Clear existing actors
    clear_existing_actors(world)

    # Generate all vehicles
    print("Starting vehicle generation...")
    for j in range(1):  # Total 100 vehicles
        if f"vehicle_{j}" in destroyed_vehicles:
            continue
        for attempt in range(3):  # Retry up to 3 times
            spawn_point = random.choice(spawn_points)
            blueprint = random.choice(vehicle_bp_library)
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicle_id = f"vehicle_{j}"
                vehicles.append((vehicle, vehicle_id))
                os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                print(f"Vehicle {vehicle_id} successfully generated in experiment {experiment_id}.")
                break
            else:
                print(f"Attempt {attempt + 1} failed to spawn vehicle {j}. Retrying...")
                time.sleep(2)

    print(f"Completed vehicle generation: {len(vehicles)} vehicles generated.")
    print("-------------------------------------------------------------")

    # Generate all walkers
    print("Starting walker generation...")
    for j in range(2):  # Total 200 walkers
        if f"walker_{j}" in destroyed_walkers:
            continue
        for attempt in range(3):
            spawn_point = world.get_random_location_from_navigation()
            if not spawn_point:
                print(f"Invalid spawn point for walker {j}, skipping.")
                break
            walker_bp = random.choice(walker_bp_library)
            walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_point))
            if walker:
                try:
                    walker_control_bp = world.get_blueprint_library().find('controller.ai.walker')
                    walker_controller = world.spawn_actor(walker_control_bp, carla.Transform(), walker)
                    walker_speed = random.uniform(1.0, 3.0)
                    walker_controller.start()
                    walker_controller.set_max_speed(walker_speed)
                    walker_controller.go_to_location(world.get_random_location_from_navigation())
                    walker_id = f"walker_{j}"
                    walkers.append((walker, walker_controller, walker_id))
                    os.makedirs(f"{walker_dir}/{walker_id}", exist_ok=True)
                    print(f"Walker {walker_id} generated with speed {walker_speed:.2f} m/s.")
                    break
                except RuntimeError as e:
                    print(f"Failed to initialize walker {walker_id}: {e}")
                    if walker.is_alive:
                        walker.destroy()

    print(f"Completed walker generation: {len(walkers)} walkers generated.")
    print("-------------------------------------------------------------")


    # Start simulation and recording
    print("Starting simulation...")
    simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, 3, destroyed_vehicles, destroyed_walkers)

    # # Clean up vehicles
    # print("Cleaning up vehicles...")
    # for vehicle, vehicle_id in vehicles:
    #     try:
    #         if vehicle.is_alive:
    #             vehicle.destroy()
    #             print(f"Vehicle {vehicle_id} successfully cleaned up.")
    #     except RuntimeError:
    #         print(f"Error cleaning up vehicle {vehicle_id}.")

    # # Clean up walkers
    # print("Cleaning up walkers...")
    # for walker, walker_controller, walker_id in walkers:
    #     try:
    #         if walker_controller.is_alive:
    #             walker_controller.stop()
    #             walker_controller.destroy()
    #             print(f"Walker controller for {walker_id} successfully cleaned up.")
    #         if walker.is_alive:
    #             walker.destroy()
    #             print(f"Walker {walker_id} successfully cleaned up.")
    #     except RuntimeError:
    #         print(f"Error cleaning up walker {walker_id}.")

    # 恢復設置
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    
    print("Simulation completed!")

def simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers):
    """
    結合同步模式與完整功能的模擬與數據記錄函數。
    確保模擬中車輛和行人保持同步，每秒記錄數據並自動補充缺失對象。
    """
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    for t in range(duration):  # 持續模擬 `duration` 秒
        world.tick()  # 同步模式下執行模擬步驟
        print(f"Simulation tick {t + 1}/{duration}")

        # 檢查車輛是否存活，若有損失則補充
        for i, (vehicle, vehicle_id) in enumerate(vehicles):
            if not vehicle.is_alive:
                print(f"Vehicle {vehicle_id} is no longer alive. Regenerating...")
                destroyed_vehicles.add(vehicle_id)

                # 嘗試生成新車輛並替代損失車輛
                for attempt in range(3):
                    try:
                        spawn_point = random.choice(spawn_points)
                        blueprint = random.choice(vehicle_bp_library)
                        new_vehicle = world.try_spawn_actor(blueprint, spawn_point)
                        if new_vehicle:
                            new_vehicle.set_autopilot(True)
                            print(f"New vehicle generated with ID {vehicle_id} at spawn point {spawn_point}.")
                            vehicles[i] = (new_vehicle, vehicle_id)
                            os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                            break
                        else:
                            print(f"Failed to spawn new vehicle for ID {vehicle_id}. Retrying...")
                            time.sleep(2)
                    except RuntimeError as e:
                        print(f"Error spawning new vehicle for ID {vehicle_id}: {e}")

        # 更新行人位置
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue

                # 移動行人
                walker_controller.go_to_location(world.get_random_location_from_navigation())
            except RuntimeError as e:
                print(f"Error updating walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

        # 捕捉車輛數據並記錄
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue
            try:
                if not vehicle.is_alive:
                    print(f"Vehicle {vehicle_id} destroyed, skipping.")
                    destroyed_vehicles.add(vehicle_id)
                    continue

                # 記錄車輛位置、速度與旋轉數據
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                print(f"Vehicle {vehicle_id} data recorded at time {t} in experiment {experiment_id}.")

                # 捕獲 RGB 與深度影像，並保存特徵點
                capture_images_with_feature_points(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t)
            except RuntimeError as e:
                print(f"Error capturing data for vehicle {vehicle_id}: {e}")
                destroyed_vehicles.add(vehicle_id)

        # 捕捉行人數據並記錄
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue

                # 記錄行人位置數據
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
                print(f"Walker {walker_id} data recorded at time {t} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error capturing data for walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

    print("Simulation completed for all time steps.")

# Process and save feature points
def process_and_save_feature_points(rgb_image, depth_image, intrinsic, transform, dirname, timestamp):
    """
    Process RGB and depth images to extract 2D and 3D feature points
    and save them to corresponding files.

    Args:
        rgb_image: The RGB image captured from the camera.
        depth_image: The depth image captured from the camera.
        intrinsic: The intrinsic matrix of the camera.
        transform: The transform of the camera in the world coordinate system.
        dirname: Directory where files will be saved.
        timestamp: Timestamp for file naming.
    """
    # print("-------process_and_save_feature_points start------")
    
    # Convert RGB image to numpy array
    rgb_array = np.array(rgb_image.raw_data).reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

    # Convert depth image to numpy array and normalize to meters
    depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    depth_array = depth_array * 1000.0 / 255.0  # Convert to meters

    # Detect 2D feature points in RGB image
    feature_points_2d = detect_feature_points(rgb_array)
    
    # Map 2D feature points to 3D world coordinates
    feature_points_3d = []
    # print(f"intrinsic: {intrinsic}")
    # print(f"transform: {transform}")
    
    for x, y in feature_points_2d:
        try:
            depth = depth_array[int(y), int(x)]
            if depth <= 0:  # Ignore invalid depth values
                continue
            pixel_coords = np.array([x, y, 1.0])
            camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
            world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
            feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])
        except Exception as e:
            print(f"Error processing feature point at ({x}, {y}): {e}")

    # Save 2D feature points
    with open(f"{dirname}/2D_FP_{timestamp}.txt", "w") as f:
        for x, y in feature_points_2d:
            f.write(f"{x} {y}\n")

    # Save 3D feature points
    with open(f"{dirname}/3D_FP_{timestamp}.txt", "w") as f:
        for x, y, z in feature_points_3d:
            f.write(f"{x} {y} {z}\n")

    # print(f"Feature points saved for timestamp {timestamp} in {dirname}.")

# Save depth image as grayscale
def save_gray_depth_image(depth_image, filename):
    """
    Save the depth image as a grayscale image.
    """
    depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    depth_normalized = (depth_array / depth_array.max() * 255).astype(np.uint8)
    cv2.imwrite(filename, depth_normalized)

# Save depth data as .ply file
def save_ply_file(depth_image, intrinsic, transform, ply_filename):
    """
    Generate a .ply file from a depth image.

    Args:
        depth_image: CARLA depth image.
        intrinsic: Camera intrinsic matrix.
        transform: Camera extrinsic transform.
        ply_filename: Path to save the .ply file.
    """
    depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    depth_array = depth_array * 1000.0 / 255.0  # Convert to meters
    points = []

    # for y in range(depth_array.shape[0]):
    #     for x in range(depth_array.shape[1]):
    #         depth = depth_array[y, x]
    #         if depth <= 0:  # Skip invalid depths
    #             continue

    #         pixel_coords = np.array([x, y, 1.0])
    #         camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
    #         world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
    #         points.append([world_coords.x, world_coords.y, world_coords.z])

    # Write to .ply file
    # with open(ply_filename, "w") as ply_file:
    #     ply_file.write("ply\n")
    #     ply_file.write("format ascii 1.0\n")
    #     ply_file.write(f"element vertex {len(points)}\n")
    #     ply_file.write("property float x\n")
    #     ply_file.write("property float y\n")
    #     ply_file.write("property float z\n")
    #     ply_file.write("end_header\n")
    #     for point in points:
    #         ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")
    height, width = depth_array.shape
    for y in range(height):
        for x in range(width):
            depth = depth_array[y, x]
            if depth <= 0:
                continue
            pixel_coords = np.array([x, y, 1.0])
            camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
            world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
            points.append(f"{world_coords.x} {world_coords.y} {world_coords.z} 255 255 255\n")

    with open(ply_filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        f.writelines(points)

    print(f"Point cloud saved as .ply at {ply_filename}.")

def capture_images_with_feature_points(world, vehicle, dirname, timestamp):
    """
    Capture RGB and depth images from a vehicle's camera, detect feature points,
    and save 2D/3D feature points and images to files.

    Args:
        world: CARLA world object.
        vehicle: Vehicle actor to attach cameras.
        dirname: Directory where outputs will be saved.
        timestamp: Timestamp for file naming.
    """
    rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
    depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

    if not rgb_camera or not depth_camera:
        print(f"Skipping capture for vehicle {dirname.split('/')[-1]} at time {timestamp}: Cameras not attached.")
        if rgb_camera:
            rgb_camera.destroy()
        if depth_camera:
            depth_camera.destroy()
        return
    
    # Containers for captured data
    rgb_image_data = []
    depth_image_data = []

    def save_rgb_image(image):
        rgb_image_data.append(image)
        # Save RGB image directly to disk
        image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")
        print(f"RGB image saved at {dirname}/rgb_{timestamp}.png.")

    def save_depth_image(image):
        """
        Save depth image as a grayscale PNG.
        """
        depth_image_data.append(image)

        # Convert CARLA depth data to numpy array
        depth_array = np.array(image.raw_data, dtype=np.float32).reshape((image.height, image.width, 4))[:, :, 0]
        depth_array_normalized = depth_array / np.max(depth_array) * 255.0  # Normalize depth to 0-255
        depth_array_normalized = depth_array_normalized.astype(np.uint8)  # Convert to 8-bit grayscale

        # Save the depth image as grayscale
        depth_image_path = f"{dirname}/depth_{timestamp}.png"
        cv2.imwrite(depth_image_path, depth_array_normalized)
        print(f"Grayscale depth image saved at {depth_image_path}.")

    rgb_camera.listen(save_rgb_image)
    depth_camera.listen(save_depth_image)

    # 等待足夠時間以確保圖像捕捉完成
    for _ in range(10):  # 最多嘗試 10 次
        world.tick()
        time.sleep(0.1)
        if rgb_image_data and depth_image_data:
            break
        else:
            print(f"Failed to capture images for vehicle {dirname.split('/')[-1]} at time {timestamp}.")
    # time.sleep(1)  # Allow time for images to be captured
    # rgb_camera.stop()
    # depth_camera.stop()

    # Precompute intrinsic and transform
    # Save grayscale depth image
    save_depth_image(depth_image_data[0])
    save_gray_depth_image(depth_image_data[0], f"{dirname}/depth_gray_{timestamp}.png")
    intrinsic = get_camera_intrinsic(rgb_camera)
    transform = rgb_camera.get_transform()

    # Process and save feature points
    process_and_save_feature_points(rgb_image_data[0], depth_image_data[0], intrinsic, transform, dirname, timestamp)

    # Save .ply file from depth image
    ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
    save_ply_file(depth_image_data[0], intrinsic, transform, ply_filename)


    # Destroy cameras
    if rgb_camera.is_alive:
        rgb_camera.destroy()
    if depth_camera.is_alive:
        depth_camera.destroy()

# Attach camera to vehicle
def attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb'):
    """
    Attach a camera of specified type to a vehicle.
    """
    try:
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")
        transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        return camera
    except RuntimeError as e:
        print(f"Failed to attach {camera_type} to vehicle: {e}")
        return None

def get_camera_intrinsic(camera):
    """
    Retrieve the intrinsic matrix of the camera.
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
    Detect feature points in the image using ORB (Oriented FAST and Rotated BRIEF).

    Args:
        image: The input image for feature detection.

    Returns:
        List of 2D feature points as (x, y) tuples.
    """
    print("--------detect_feature_points------------")
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(gray, None)
    return [kp.pt for kp in keypoints]

def simulate_and_record_sync(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration):
    """
    在同步模式下運行模擬，確保每秒統一暫停並捕捉數據。
    """
    for t in range(duration):  # 模擬總時長為 `duration` 秒
        world.tick()  # 執行模擬一步，固定步長為 1 秒
        print(f"Simulation tick {t + 1}/{duration}")

        # 捕捉車輛數據
        for vehicle, vehicle_id in vehicles:
            if not vehicle.is_alive:
                print(f"Vehicle {vehicle_id} destroyed, skipping.")
                continue
            try:
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                print(f"Vehicle {vehicle_id} data recorded.")
                capture_images_with_feature_points(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t)
            except RuntimeError as e:
                print(f"Error capturing data for vehicle {vehicle_id}: {e}")

        # 捕捉行人數據
        for walker, walker_controller, walker_id in walkers:
            if not walker.is_alive:
                print(f"Walker {walker_id} destroyed, skipping.")
                continue
            try:
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
                print(f"Walker {walker_id} data recorded.")
            except RuntimeError as e:
                print(f"Error capturing data for walker {walker_id}: {e}")

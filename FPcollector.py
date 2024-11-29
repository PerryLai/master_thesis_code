import carla
import os
import sys
import time
import random
import cv2
import numpy as np
import tqdm

# 單純是我不想一直看到一堆WARNING所作的措施
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

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

    print(f"已清除 {len(vehicles)} 輛車輛、{len(walkers)} 名行人，以及 {len(controllers)} 個控制器。")

def generate_simulation_data(world, experiment_id, dirname, duration, vehicle_num, walker_num):

    # 設置同步模擬模式
    sys.stderr = open(os.devnull, 'w')
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0
    settings.max_substep_delta_time = 0.05
    settings.max_substeps = 20
    world.apply_settings(settings)
    sys.stderr = sys.__stderr__

    # 嘗試多次應用設置
    for attempt in range(10):
        world.apply_settings(settings)
        updated_settings = world.get_settings()
        if updated_settings.synchronous_mode and updated_settings.fixed_delta_seconds == 1.0:
            # print("Settings applied successfully.")
            # print("設置已成功應用。")
            break
        time.sleep(0.1)  # 短暫等待重試
    else:
        print("在嘗試 30 次後仍無法成功應用設置。")

    # 生成車輛和行人數據，並記錄它們的數據，還有捕獲特徵點的影像。
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    walker_bp_library = world.get_blueprint_library().filter('*walker*')
    spawn_points = world.get_map().get_spawn_points()

    # 建立目錄
    vehicle_dir = os.path.join(dirname, "vehicle")
    walker_dir = os.path.join(dirname, "walkers")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(walker_dir, exist_ok=True)

    destroyed_vehicles = set()
    destroyed_walkers = set()

    vehicles, walkers = [], []

    # 清除已存在的角色
    clear_existing_actors(world)

    # 生成所有車輛
    print("開始生成車輛...")
    for j in range(vehicle_num):  # 總共生成 100 輛車
        if f"vehicle_{j}" in destroyed_vehicles:
            continue
        for attempt in range(10):  # 最多重試 10 次
            spawn_point = random.choice(spawn_points)
            blueprint = random.choice(vehicle_bp_library)
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicle_id = f"vehicle_{j}"
                vehicles.append((vehicle, vehicle_id))
                os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                print(f"車輛 {vehicle_id} 成功生成於實驗 {experiment_id} 中。")
                break
            else:
                print(f"嘗試第 {attempt + 1} 次生成車輛 {j} 失敗，重試中...")
                time.sleep(2)

    print(f"---------車輛生成完成：共生成 {len(vehicles)} 輛車輛。------------")

    # 生成所有行人
    print("開始生成行人...")
    for j in range(walker_num):  # 總共生成 200 名行人
        if f"walker_{j}" in destroyed_walkers:
            continue
        for attempt in range(3):
            spawn_point = world.get_random_location_from_navigation()
            if not spawn_point:
                print(f"行人 {j} 的生成點無效，跳過。")
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
                    # print(f"Walker {walker_id} generated with speed {walker_speed:.2f} m/s.")
                    # print(f"行人 {walker_id} 生成，速度為 {walker_speed:.2f} 米/秒。")
                    break
                except RuntimeError as e:
                    print(f"初始化行人 {walker_id} 失敗：{e}")
                    if walker.is_alive:
                        walker.destroy()

    print(f"---------行人生成完成：共生成 {len(walkers)} 名行人。---------")

    # 開始模擬並記錄數據
    print("---------開始模擬...---------")
    simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers)

    # 恢復設置
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("模擬完成！")

def simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers):
    """
    結合同步模式與完整功能的模擬與數據記錄函數。
    確保模擬中車輛和行人保持同步，每秒記錄數據並自動補充缺失對象。
    """
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    fp_town_points = "output/PCD_FP_town01.txt"

    for t in range(duration):  # 持續模擬 `duration` 秒
        world.tick()  # 同步模式下執行模擬步驟
        print(f"---------模擬步驟 {t + 1}/{duration}---------")

        # 檢查車輛是否存活，若有損失則補充
        for i, (vehicle, vehicle_id) in enumerate(vehicles):
            if not vehicle.is_alive:
                print(f"車輛 {vehicle_id} 已不再存活，重新生成中...")
                destroyed_vehicles.add(vehicle_id)

                # 嘗試生成新車輛並替代損失車輛
                for attempt in range(3):
                    try:
                        spawn_point = random.choice(spawn_points)
                        blueprint = random.choice(vehicle_bp_library)
                        new_vehicle = world.try_spawn_actor(blueprint, spawn_point)
                        if new_vehicle:
                            new_vehicle.set_autopilot(True)
                            print(f"新的車輛 {vehicle_id} 已於生成點 {spawn_point} 生成。")
                            vehicles[i] = (new_vehicle, vehicle_id)
                            os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                            break
                        else:
                            print(f"嘗試生成車輛 {vehicle_id} 失敗，重試中...")
                            time.sleep(2)
                    except RuntimeError as e:
                        print(f"生成車輛 {vehicle_id} 時發生錯誤：{e}")

        # 更新行人位置
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                if not walker.is_alive:
                    print(f"行人 {walker_id} 已被摧毀，跳過處理。")
                    destroyed_walkers.add(walker_id)
                    continue

                # 移動行人
                walker_controller.go_to_location(world.get_random_location_from_navigation())
            except RuntimeError as e:
                print(f"更新行人 {walker_id} 時發生錯誤：{e}")
                destroyed_walkers.add(walker_id)

        # 捕捉車輛數據並記錄
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue
            try:
                if not vehicle.is_alive:
                    print(f"車輛 {vehicle_id} 已被摧毀，跳過處理。")
                    destroyed_vehicles.add(vehicle_id)
                    continue

                # 記錄車輛位置、速度與旋轉數據，取整數值
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"時間 {t}: 位置: ({int(location.x)}, {int(location.y)}, {int(location.z)}), "
                            f"速度: ({int(velocity.x)}, {int(velocity.y)}, {int(velocity.z)}), "
                            f"旋轉: ({int(rotation.pitch)}, {int(rotation.yaw)}, {int(rotation.roll)})\n")
                print(f"已於時間 {t} 記錄車輛 {vehicle_id} 的數據，屬於實驗 {experiment_id}。")

                # 捕獲 RGB 與深度影像，並保存特徵點
                capture_images_with_feature_points(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t, fp_town_points)
            except RuntimeError as e:
                print(f"捕捉車輛 {vehicle_id} 數據時發生錯誤：{e}")
                destroyed_vehicles.add(vehicle_id)

        # 捕捉行人數據並記錄
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                if not walker.is_alive:
                    print(f"行人 {walker_id} 已被摧毀，跳過處理。")
                    destroyed_walkers.add(walker_id)
                    continue

                # 記錄行人位置數據，取整數值
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"時間 {t}: 位置: ({int(location.x)}, {int(location.y)}, {int(location.z)})\n")
                print(f"已於時間 {t} 記錄行人 {walker_id} 的數據，屬於實驗 {experiment_id}。")
            except RuntimeError as e:
                print(f"捕捉行人 {walker_id} 數據時發生錯誤：{e}")
                destroyed_walkers.add(walker_id)

    print("所有時間步驟的模擬已完成。")

def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file):
    """
    捕捉 RGB 和深度影像，將 2D 特徵點轉換為 3D 世界座標，
    並與 PCD_FP_town01.txt 中的點進行匹配。

    Args:
        world: CARLA 世界對象。
        vehicle: 要附加相機的車輛。
        dirname: 輸出檔案保存的目錄。
        timestamp: 檔案命名的時間戳。
        fp_town_file: 包含世界座標的 PCD_FP_town01.txt 的路徑。
    """
    # 附加相機
    rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
    depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

    if not rgb_camera or not depth_camera:
        print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加相機。")
        if rgb_camera:
            rgb_camera.destroy()
        if depth_camera:
            depth_camera.destroy()
        return

    # 加載 PCD_FP_town01.txt 中的點
    try:
        fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
    except Exception as e:
        print(f"載入 PCD_FP_town01.txt 錯誤：{e}")
        return

    # 用於捕捉數據的容器
    rgb_image_data = []
    depth_image_data = []

    def save_rgb_image(image):
        rgb_image_data.append(image)
        image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")
        # print(f"RGB 影像已保存於 {dirname}/rgb_{timestamp}.png。")

    def save_depth_image(image):
        depth_image_data.append(image)
        depth_array = np.array(image.raw_data, dtype=np.float32).reshape((image.height, image.width, 4))[:, :, 0]
        depth_array_normalized = (depth_array / np.max(depth_array) * 255.0).astype(np.uint8)
        depth_image_path = f"{dirname}/depth_{timestamp}.png"
        cv2.imwrite(depth_image_path, depth_array_normalized)
        # print(f"灰階深度影像已保存於 {depth_image_path}。")

    rgb_camera.listen(save_rgb_image)
    depth_camera.listen(save_depth_image)

    # 等待影像捕捉完成
    for _ in range(20):
        world.tick()
        time.sleep(0.1)
        if rgb_image_data and depth_image_data:
            break

    # 停止監聽相機
    rgb_camera.stop()
    depth_camera.stop()

    if not (rgb_image_data and depth_image_data):
        print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
    else:
        # 預計算相機內參與轉換
        intrinsic = get_camera_intrinsic(rgb_camera)
        transform = rgb_camera.get_transform()

        # 在 RGB 影像中檢測特徵點
        rgb_array = np.array(rgb_image_data[0].raw_data).reshape((720, 1280, 4))[:, :, :3]
        detected_fp = detect_feature_points(rgb_array)

        # 將 2D 特徵點轉換為 3D 世界座標
        feature_points_3d = []
        depth_array = np.array(depth_image_data[0].raw_data, dtype=np.float32).reshape((720, 1280, 4))[:, :, 0]

        for x, y in detected_fp:
            try:
                depth = depth_array[int(y), int(x)]
                if depth <= 0:
                    continue  # 跳過無效深度點

                # 轉換為相機座標
                pixel_coords = np.array([x, y, 1.0])
                camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth

                # 轉換為世界座標
                world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
                feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])
            except Exception as e:
                print(f"將 2D 特徵點 ({x}, {y}) 轉換為 3D 世界座標時出錯：{e}")

        feature_points_3d = np.array(feature_points_3d)

        # 與 PCD_FP_town01.txt 中的點進行匹配
        match_count = sum(
            1 for wp in feature_points_3d
            if any(np.linalg.norm(wp - tp) < 5 for tp in fp_town_points)
        )

        # 保存每張影像的 3D 特徵點
        np.savetxt(f"{dirname}/3D_FP_{timestamp}.txt", feature_points_3d, fmt="%.6f", delimiter=" ")
        print(f"3D 特徵點已保存於 {dirname}/3D_FP_{timestamp}.txt。")

        if match_count >= 4:
            # 輸出匹配數量
            print("")
            print(f"影像滿足條件!{dirname.split('/')[-1]}: 時間 {timestamp} 的影像匹配了 {match_count} 個 PCD_FP_town01.txt 中的點，開始處理數據。")
            print("")

            # 保存灰階深度影像
            save_depth_image(depth_image_data[0])

            # 處理並保存特徵點
            process_and_save_feature_points(rgb_image_data[0], depth_image_data[0], intrinsic, transform, dirname, timestamp)

            # 從深度影像生成並保存 .ply 文件
            ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
            save_ply_file(depth_image_data[0], intrinsic, transform, ply_filename)

            # 估計相機姿態
            try:
                detected_fp_array = np.array(detected_fp, dtype=np.float32)
                fp_town_points_array = np.array(fp_town_points[:len(detected_fp)], dtype=np.float32)
                rvec, tvec = estimate_camera_pose_2d_3d(detected_fp_array, fp_town_points_array)

                with open(f"{dirname}/pose_{timestamp}.txt", "w") as pose_file:
                    pose_file.write(f"旋轉向量：{rvec.tolist()}\n")
                    pose_file.write(f"平移向量：{tvec.tolist()}\n")
                print(f"相機姿態已保存於 {dirname}/pose_{timestamp}.txt。")
            except Exception as e:
                print(f"估計相機姿態時出錯：{e}")
        else:
            print(f"{dirname.split('/')[-1]}: 時間 {timestamp} 的影像不滿足條件，跳過。")

    # 銷毀相機
    if rgb_camera.is_alive:
        rgb_camera.destroy()
    if depth_camera.is_alive:
        depth_camera.destroy()

# Process and save feature points
def process_and_save_feature_points(rgb_image, depth_image, intrinsic, transform, dirname, timestamp):
    """
    處理 RGB 和深度影像以提取 2D 和 3D 特徵點，並將結果保存到對應檔案。

    Args:
        rgb_image: 從相機捕獲的 RGB 影像。
        depth_image: 從相機捕獲的深度影像。
        intrinsic: 相機的內部參數矩陣。
        transform: 相機在世界座標系中的變換矩陣。
        dirname: 保存檔案的目錄。
        timestamp: 檔案命名的時間戳。
    """
    # 將 RGB 影像轉換為 numpy 陣列
    rgb_array = np.array(rgb_image.raw_data).reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

    # 將深度影像轉換為 numpy 陣列並歸一化為公尺
    depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    depth_array = depth_array * 1000.0 / 255.0  # 轉換為公尺

    # 在 RGB 影像中檢測 2D 特徵點
    feature_points_2d = detect_feature_points(rgb_array)

    # 將 2D 特徵點映射到 3D 世界座標
    feature_points_3d = []
    for x, y in feature_points_2d:
        try:
            depth = depth_array[int(y), int(x)]
            if depth <= 0:  # 忽略無效的深度值
                continue
            pixel_coords = np.array([x, y, 1.0])
            camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
            world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
            feature_points_3d.append([int(world_coords.x), int(world_coords.y), int(world_coords.z)])  # 取整數
        except Exception as e:
            print(f"處理特徵點 ({x}, {y}) 時發生錯誤：{e}")

    # 保存 2D 特徵點
    with open(f"{dirname}/2D_FP_{timestamp}.txt", "w") as f:
        for x, y in feature_points_2d:
            f.write(f"{int(x)} {int(y)}\n")  # 取整數後保存

    # 保存 3D 特徵點
    with open(f"{dirname}/3D_FP_{timestamp}.txt", "w") as f:
        for x, y, z in feature_points_3d:
            f.write(f"{x} {y} {z}\n")  # 3D 特徵點已取整數

    print(f"特徵點已保存至 {dirname}，時間戳：{timestamp}。")

# Save depth image as grayscale
def save_gray_depth_image(depth_image, filename):
    """
    將深度影像保存為灰階影像。
    """
    depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    depth_normalized = (depth_array / depth_array.max() * 255).astype(np.uint8)
    cv2.imwrite(filename, depth_normalized)
    print(f"深度影像已保存為灰階圖像：{filename}")

# Save depth data as .ply file
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


def attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb'):
    """
    將指定類型的相機附加到車輛上。
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
        print(f"將 {camera_type} 附加到車輛失敗：{e}")
        return None

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

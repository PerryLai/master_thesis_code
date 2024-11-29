import os
import random
import time
import numpy as np
import carla
import sys
import cv2
from pcd_to_ply import process_all_ply_files
from util import clear_existing_actors, save_pointcloud_to_ply, attach_camera_to_vehicle, get_camera_intrinsic
from tqdm import tqdm

def generate_simulation_data(world, experiment_id, dirname, duration, vehicle_num, walker_num, select_num, fp_town_file, fp_town_new_file):

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
                    print(f"行人 {walker_id} 生成，速度為 {walker_speed:.2f} 米/秒。")
                    break
                except RuntimeError as e:
                    print(f"初始化行人 {walker_id} 失敗：{e}")
                    if walker.is_alive:
                        walker.destroy()

    print(f"---------行人生成完成：共生成 {len(walkers)} 名行人。---------")

    # 開始模擬並記錄數據
    print("---------開始模擬...---------")
    simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers, select_num, fp_town_file, fp_town_new_file)

    # 刪除空目錄
    print("---------開始刪除空目錄...---------")
    for root, dirs, files in os.walk(dirname, topdown=False):  # 自底向上遍歷
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):  # 如果目錄是空的
                os.rmdir(dir_path)
                # print(f"已刪除空目錄：{dir_path}")
    print("空目錄刪除完成！")

    process_all_ply_files(dirname)

    # 恢復設置
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("模擬完成！")

def simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers, select_num, fp_town_file, fp_town_new_file):
    """
    結合同步模式與完整功能的模擬與數據記錄函數。
    確保模擬中車輛和行人保持同步，每秒記錄數據並自動補充缺失對象。
    """
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    fp_town_points = "output/FP_town01.txt"
    print("")
    print(f"---------實驗 {experiment_id} 開始---------")
    print("")

    # 隨機選取 5 台車輛進行檢測
    if len(vehicles) > select_num:
        selected_vehicles = random.sample(vehicles, select_num)  # 隨機選擇 5 台
    else:
        selected_vehicles = vehicles  # 如果車輛數量少於或等於 5，則全部檢測

    for t in range(duration):  # 持續模擬 `duration` 秒
        world.tick()  # 同步模式下執行模擬步驟
        print(f"---------模擬秒數 {t + 1}/{duration}---------")

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
        for vehicle, vehicle_id in selected_vehicles:
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
                print("")
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"時間 {t}: 位置: ({location.x:.0f}, {location.y:.0f}, {location.z:.0f}), "
                            f"速度: ({velocity.x:.0f}, {velocity.y:.0f}, {velocity.z:.0f}), "
                            f"旋轉: ({rotation.pitch:.0f}, {rotation.yaw:.0f}, {rotation.roll:.0f})\n")
                print(f"已於時間 {t} 記錄車輛 {vehicle_id} 的整數值數據。")
                print(f"時間 {t}: 位置: ({location.x:.0f}, {location.y:.0f}, {location.z:.0f}), "
                            f"速度: ({velocity.x:.0f}, {velocity.y:.0f}, {velocity.z:.0f}), "
                            f"旋轉: ({rotation.pitch:.0f}, {rotation.yaw:.0f}, {rotation.roll:.0f})\n")
                with open(f"{vehicle_dir}/{vehicle_id}/info_int.txt", "a") as f:
                    f.write(f"時間 {t}: 位置: ({location.x:.3f}, {location.y:.3f}, {location.z:.3f}), "
                            f"速度: ({velocity.x:.3f}, {velocity.y:.3f}, {velocity.z:.3f}), "
                            f"旋轉: ({rotation.pitch:.3f}, {rotation.yaw:.3f}, {rotation.roll:.3f})\n")
                print(f"已於時間 {t} 記錄車輛 {vehicle_id} 的浮點數值數據。")
                print(f"時間 {t}: 位置: ({location.x:.3f}, {location.y:.3f}, {location.z:.3f}), "
                            f"速度: ({velocity.x:.3f}, {velocity.y:.3f}, {velocity.z:.3f}), "
                            f"旋轉: ({rotation.pitch:.3f}, {rotation.yaw:.3f}, {rotation.roll:.3f})\n")

                # with open(f"{vehicle_dir}/{vehicle_id}/info_int.txt", "a") as f:
                #     f.write(f"時間 {t}: 位置: ({int(location.x)}, {int(location.y)}, {int(location.z)}), "
                #             f"速度: ({int(velocity.x)}, {int(velocity.y)}, {int(velocity.z)}), "
                #             f"旋轉: ({int(rotation.pitch)}, {int(rotation.yaw)}, {int(rotation.roll)})\n")
                # print(f"已於時間 {t} 記錄車輛 {vehicle_id} 的整數值數據。")
                # with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                #     f.write(f"時間 {t}: 位置: ({format(location.x, '.3f')}, {format(location.y, '.3f')}, {format(location.z, '.3f')}), "
                #             f"速度: ({format(velocity.x, '.3f')}, {format(velocity.y, '.3f')}, {format(velocity.z, '.3f')}), "
                #             f"旋轉: ({format(rotation.pitch, '.3f')}, {format(rotation.yaw, '.3f')}, {format(rotation.roll, '.3f')})\n")
                # print(f"已於時間 {t} 記錄車輛 {vehicle_id} 的浮點數值數據。")

                # 捕獲 RGB 與深度影像，並保存特徵點
                capture_images_with_feature_points(experiment_id, world, vehicle, f"{vehicle_dir}/{vehicle_id}", t, fp_town_file, fp_town_new_file, t, vehicle_id)
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
                print(f"已於時間 {t} 記錄行人 {walker_id} 的數據。")
            except RuntimeError as e:
                print(f"捕捉行人 {walker_id} 數據時發生錯誤：{e}")
                destroyed_walkers.add(walker_id)

    print("所有時間步驟的模擬已完成。")

def capture_images_with_feature_points(experiment_id, world, vehicle, dirname, timestamp, fp_town_file, fp_town_new_file, t, vehicle_id):
    """
    捕捉 RGB 和深度影像，並生成點雲（深度影像和 LiDAR）。
    """
    # 附加相機
    rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
    depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

    # 附加 LiDAR
    # lidar = attach_lidar_to_vehicle(world, vehicle)

    # if not rgb_camera or not depth_camera or not lidar:
    if not rgb_camera or not depth_camera:
        print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加感測器。")
        if rgb_camera:
            rgb_camera.destroy()
        if depth_camera:
            depth_camera.destroy()
        # if lidar:
        #     lidar.destroy()
        return
    
    # 加載 FP_town01.txt 中的點
    try:
        fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
    except Exception as e:
        print(f"載入 {fp_town_file} 錯誤：{e}")
        return

    # 用於捕捉數據的容器
    rgb_image_data = []
    depth_image_data = []
    lidar_data = None

    def save_rgb_image(image):
        rgb_image_data.append(image)

    def save_depth_image(image):
        depth_image_data.append(image)

    # def lidar_callback(data):
    #     nonlocal lidar_data
    #     lidar_data = data

    rgb_camera.listen(save_rgb_image)
    depth_camera.listen(save_depth_image)

    # if not lidar:
    #     print("LiDAR 感測器未成功附加到車輛。")
    #     return

    # if not lidar.is_listening:
    #     print("LiDAR 感測器尚未啟動，準備啟動...")

    # # 啟用 LiDAR 數據監聽
    # time.sleep(1)  # 等待 1 秒確保感測器啟動
    # lidar.listen(lidar_callback)

    # 等待影像捕捉完成
    for _ in range(20):
        world.tick()
        time.sleep(0.1)
        if rgb_image_data and depth_image_data and lidar_data:
            break

    # 停止監聽相機
    rgb_camera.stop()
    depth_camera.stop()

    # if lidar.is_listening:
    #     lidar.stop()
    # else:
    #     print("LiDAR 感測器未啟動，無需停止。")

    if not (rgb_image_data and depth_image_data):
        print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
    else:
        # 原始深度影像處理
        depth_image = depth_image_data[0]
        depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]

        # 增強深度數據對比
        depth_array_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        depth_array_normalized *= 3000.0  # 調整至有效深度範圍（例如 0-1000 米）

        # # 使用 Matplotlib 顯示深度影像，3 秒後自動關閉
        # plt.imshow(depth_array_normalized, cmap='gray')
        # plt.title("Depth Image")
        # plt.axis('off')
        # plt.draw()  # 繪製影像
        # plt.pause(3)  # 停留 3 秒
        # plt.close()  # 關閉視窗

        # 繼續執行後續邏輯
        intrinsic = get_camera_intrinsic(depth_camera)
        transform = depth_camera.get_transform()

        if np.all(depth_array_normalized <= 0):
            print("深度影像未包含有效數據，點雲生成失敗。")
            return      

        points = []

        matched_points = []  # 完整小數座標
        matched_int_set = set()  # 用於檢查整數座標是否已匹配過

        for y in range(depth_array.shape[0]):
            for x in range(depth_array.shape[1]):
                depth = depth_array[y, x]
                if depth <= 0:
                    continue
                # 將像素點轉換為世界座標
                pixel_coords = np.array([x, y, 1.0])
                camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
                world_coords = transform.transform(carla.Location(
                    x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]
                ))
                points.append([world_coords.x, world_coords.y, world_coords.z])  # 將完整的小數座標加入點雲
                # 匹配點
                int_point = (int(world_coords.x), int(world_coords.y), int(world_coords.z))

                # 檢查是否已經匹配過
                if int_point in matched_int_set:
                    continue

                # 比對 FP_town01.txt
                distances = np.linalg.norm(fp_town_points - np.array(int_point), axis=1)
                if np.any(distances < 0.01):  # 距離閾值
                    matched_points.append([world_coords.x, world_coords.y, world_coords.z])
                    matched_int_set.add(int_point)  # 記錄整數值避免重複

        # 確認 FP 數量是否大於 4
        if len(matched_points) > 4:
            print("")
            print(f"這張圖適合當KF！影像匹配了 {len(matched_points)} 個特徵點，開始保存數據...")
            print("")
            
            # 保存到全局文件 fp_town_new_file.txt
            with open(fp_town_new_file, "a") as f:
                for int_point in matched_int_set:
                    f.write(f"{int_point[0]} {int_point[1]} {int_point[2]}\n")

            print(f"成功保存匹配點至 {fp_town_new_file}")

            # 保存灰階深度影像
            depth_array_gray = (depth_array_normalized / depth_array_normalized.max() * 255).astype(np.uint8)
            cv2.imwrite(f"{dirname}/depth_{timestamp}.png", depth_array_gray)
            print(f"深度影像已保存至 {dirname}/depth_{timestamp}.png。")

            # 保存 RGB 影像
            rgb_image = rgb_image_data[0]
            rgb_image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")
            print(f"RGB影像已保存至{dirname}/rgb_{timestamp}.png。")

            # 保存基於深度影像生成的點雲
            ply_filename = f"{dirname}/pointcloud_depth_{timestamp}.ply"
            save_pointcloud_to_ply(points, ply_filename)
            print(f"基於深度影像生成的點雲已保存至{dirname}/pointcloud_{timestamp}.ply。")

            # # LiDAR 點雲處理
            # lidar_ply_filename = f"{dirname}/pointcloud_lidar_{timestamp}.ply"
            # capture_lidar_pointcloud(lidar, lidar_ply_filename)
            # print(f"基於LiDAR的點雲已保存至{dirname}/pointcloud_{timestamp}.ply。")

            # 保存完整小數座標特徵點
            matched_txt_filename = f"{dirname}/FP_{timestamp}.txt"
            np.savetxt(matched_txt_filename, matched_points, fmt="%.6f", delimiter=" ")
            print(f"完整特徵點已保存於 {matched_txt_filename}。")

            # 保存去重的整數座標特徵點
            matched_int_filename = f"{dirname}/FP_int_{timestamp}.txt"
            matched_int_points = np.array(list(matched_int_set))  # 將 set 轉為 numpy 陣列
            np.savetxt(matched_int_filename, matched_int_points, fmt="%d", delimiter=" ")
            print(f"整數特徵點已保存於 {matched_int_filename}。")

            # 新增 EPnP 驗證
            validate_epnp(vehicle, matched_points, dirname, timestamp, intrinsic, experiment_id, t, vehicle_id)
        else:
            print(f"時間 {timestamp} 的影像不滿足條件，跳過保存。")

    # 銷毀感測器
    if rgb_camera.is_alive:
        rgb_camera.destroy()
    if depth_camera.is_alive:
        depth_camera.destroy()
    # if lidar.is_alive:
    #     lidar.destroy()

def capture_lidar_pointcloud(lidar, output_ply_path, timeout=5):
    """
    使用 LiDAR 感測器生成點雲並保存為 PLY 文件，添加進度條顯示和超時邏輯。
    """
    lidar_data = None

    def lidar_callback(data):
        nonlocal lidar_data
        lidar_data = data

    # 開始監聽 LiDAR 數據
    lidar.listen(lidar_callback)

    try:
        print("等待 LiDAR 數據...")
        start_time = time.time()

        # 等待 LiDAR 數據的非阻塞處理
        while lidar_data is None:
            if time.time() - start_time > timeout:
                print(f"start_time: {start_time}")
                print(f"time.time(): {time.time()}")
                print("等待 LiDAR 數據超時。")
                lidar.stop()
                return
            time.sleep(0.1)

        # 提取 LiDAR 數據大小
        print("LiDAR 數據捕獲完成，開始解析...")
        raw_data = lidar_data.raw_data
        total_points = len(raw_data) // (4 * 4)  # 每點由 4 個 float32 組成
        points = []

        print(f"解析 {total_points} 個點的 LiDAR 數據...")

        # 使用進度條逐步處理數據
        for i in tqdm(range(0, len(raw_data), 4 * 4), desc="Processing LiDAR Data"):
            chunk = raw_data[i:i + 4 * 4]
            if len(chunk) < 16:  # 確保分段長度足夠
                break
            point = np.frombuffer(chunk, dtype=np.float32)[:3]
            points.append(point)

        # 保存為 PLY 文件
        with open(output_ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(f"LiDAR 點雲已保存至 {output_ply_path}")

    finally:
        lidar.stop()

def validate_epnp(vehicle, matched_points, dirname, timestamp, intrinsic, experiment_id, t, vehicle_id):
    """
    驗證 EPnP 演算法是否正確，並統一世界座標系。
    """
    if len(matched_points) < 4:
        print(f"時間 {timestamp}: 特徵點不足，無法進行 EPnP 驗證。")
        return

    # 當前車輛的真實位置和方向
    location = vehicle.get_location()
    rotation = vehicle.get_transform().rotation
    # vehicle_position = np.array([location.x, location.y, location.z])
    vehicle_position = np.array([round(location.x, 3), round(location.y, 3), round(location.z, 3)], dtype=np.float64)
    vehicle_direction = np.array([
        np.cos(np.radians(rotation.yaw)),
        np.sin(np.radians(rotation.yaw)),
        0
    ])

    print(f"vehicle_position真實值: {vehicle_position}")
    print(f"vehicle_direction真實值: {vehicle_direction}")

    # 擷取 2D 和 3D 點
    image_points = np.array([pt[:2] for pt in matched_points], dtype=np.float32)
    world_points = np.array(matched_points, dtype=np.float32)

    # 計算 EPnP
    try:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=world_points,
            imagePoints=image_points,
            cameraMatrix=intrinsic,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            raise ValueError("EPnP 解算失敗")

        # 相機到世界的旋轉和平移矩陣
        R_cam_to_world, _ = cv2.Rodrigues(rvec)
        t_cam_to_world = tvec.ravel()
        print(f"R_cam_to_world: {R_cam_to_world}")
        print(f"t_cam_to_world: {t_cam_to_world}")

        # EPnP 的位置和方向轉換至世界座標系
        epnp_position_camera = -np.dot(R_cam_to_world.T, t_cam_to_world)
        epnp_position_world = np.dot(R_cam_to_world, epnp_position_camera) + t_cam_to_world

        print(f"epnp_position_camera: {epnp_position_camera}")
        print(f"epnp_position (世界坐標): {epnp_position_world}")

        # 計算誤差
        position_error = np.linalg.norm(epnp_position_world - vehicle_position)
        print("")
        print(f"EPnP 位置誤差: {position_error:.3f} 米")
        with open(f"output/EPnP位置誤差結果.txt", "a") as f:
                    f.write(f"{vehicle_id}: 第{experiment_id}次實驗的第{t}秒: EPnP 位置誤差: {position_error:.3f} 米\n")
        print("")

    except Exception as e:
        print(f"EPnP 驗證失敗：{e}")


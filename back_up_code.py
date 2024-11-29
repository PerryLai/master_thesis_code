# city_camera_set: 在town中心&東西南北各放置一顆鏡頭拍攝中間點，模擬低軌道衛星的拍攝方式
def city_camera_set(world, center, dirname, high=500000):  # 高度500km
    """
    設置城市中心及四個方向（東、南、西、北）的相機，生成深度和RGB影像，並存儲為點雲。
    """
    camera_list_city = []
    offset = 5000  # 5km 偏移，確保城市尺寸為 10km x 10km
    directions = [
        {"name": "center", "location": (0, 0, high), "rotation": (90, 0, 0)},
        {"name": "north", "location": (0, offset, high), "rotation": (90, 0, 0)},
        {"name": "east", "location": (offset, 0, high), "rotation": (90, 90, 0)},
        {"name": "south", "location": (0, -offset, high), "rotation": (90, 180, 0)},
        {"name": "west", "location": (-offset, 0, high), "rotation": (90, -90, 0)}
    ]
    
    for direction in directions:
        # RGB相機設置
        rgb_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_bp.set_attribute("image_size_x", str(10000))  # 高分辨率設定
        rgb_bp.set_attribute("image_size_y", str(10000))
        rgb_bp.set_attribute("fov", str(120))  # 視野範圍
        
        rgb_location = carla.Location(*direction["location"])
        rgb_rotation = carla.Rotation(pitch=direction["rotation"][0], yaw=direction["rotation"][1], roll=direction["rotation"][2])
        rgb_transform = carla.Transform(rgb_location, rgb_rotation)
        
        rgb_cam = world.spawn_actor(rgb_bp, rgb_transform)
        rgb_cam.listen(lambda image, name=direction["name"]: image.save_to_disk(f"{dirname}/{name}_rgb.jpg"))
        camera_list_city.append(rgb_cam)
        
        # 深度相機設置
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("fov", str(120))
        
        depth_transform = carla.Transform(rgb_location, rgb_rotation)
        depth_cam = world.spawn_actor(depth_bp, depth_transform)
        
        def save_depth(image, name=direction["name"]):
            point_cloud = depth_to_local_point_cloud(image, max_depth=1000, high=high)  # 最大深度1000米
            image.save_to_disk(f"{dirname}/{name}_depth.jpg", carla.ColorConverter.LogarithmicDepth)
            point_cloud.save_to_disk(f"{dirname}/{name}.ply")
        
        depth_cam.listen(save_depth)
        camera_list_city.append(depth_cam)

    return camera_list_city


import carla
import os
import time
import random


def generate_simulation_data(world, experiment_id, dirname, duration=10, walker_batch_size=100, vehicle_batch_size=50):
    """
    Generate vehicles and walkers, record their data, and capture images with feature points.
    """
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    walker_bp_library = world.get_blueprint_library().filter('*walker*')
    spawn_points = world.get_map().get_spawn_points()

    # Create directories
    vehicle_dir = os.path.join(dirname, "vehicle")
    walker_dir = os.path.join(dirname, "walkers")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(walker_dir, exist_ok=True)

    # 記錄不存在的目標
    destroyed_vehicles = set()
    destroyed_walkers = set()

    for batch_id in range(0, 1000, max(vehicle_batch_size, walker_batch_size)):
        vehicles, walkers = [], []

        # Generate vehicles
        for j in range(batch_id, min(batch_id + vehicle_batch_size, 1000)):
            if f"vehicle_{j}" in destroyed_vehicles:
                continue  # 跳過已確認不存在的目標
            spawn_point = random.choice(spawn_points)
            blueprint = random.choice(vehicle_bp_library)
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicle_id = f"vehicle_{j}"
                vehicles.append((vehicle, vehicle_id))
                os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                print(f"Vehicle {j} generated in experiment {experiment_id}.")

        # Generate walkers
        for j in range(batch_id, min(batch_id + walker_batch_size, 1000)):
            if f"walker_{j}" in destroyed_walkers:
                continue  # 跳過已確認不存在的目標
            spawn_point = world.get_random_location_from_navigation()
            if not spawn_point:
                print(f"Invalid spawn point for walker {j}, skipping.")
                continue

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
                    print(f"Walker {j} generated with speed {walker_speed:.2f} m/s.")
                except RuntimeError as e:
                    print(f"Failed to initialize walker {j}: {e}")
                    if walker.is_alive:
                        walker.destroy()

        # Simulate and record
        simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers)

        # Clean up vehicles
        for vehicle, vehicle_id in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        # Clean up walkers
        for walker, walker_controller, walker_id in walkers:
            if walker_controller.is_alive:
                walker_controller.stop()
                walker_controller.destroy()
            if walker.is_alive:
                walker.destroy()

        print(f"Batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1} completed.")

    print("All simulation batches completed!")

def simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers):
    """
    Simulate vehicles and walkers, record their movements, and capture images.
    """
    for t in range(duration):
        world.tick()
        time.sleep(1)

        # Record vehicle data
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue  # Skip destroyed vehicles
            try:
                if not vehicle.is_alive:
                    print(f"Vehicle {vehicle_id} destroyed, skipping.")
                    destroyed_vehicles.add(vehicle_id)
                    continue
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                capture_single_vehicle_image(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t)
            except RuntimeError as e:
                print(f"Error operating on vehicle {vehicle_id}: {e}")
                destroyed_vehicles.add(vehicle_id)

        # Record walker data
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue  # Skip destroyed walkers
            try:
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
            except RuntimeError as e:
                print(f"Error operating on walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

    print(f"Time {t}: Simulation step completed.")

# def simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers):
#     """
#     Simulate vehicles and walkers, record their movements, and capture images.
#     """
#     for t in range(duration):
#         world.tick()
#         time.sleep(1)

#         # Record vehicle data
#         for vehicle, vehicle_id in vehicles:
#             if vehicle_id in destroyed_vehicles:
#                 continue  # 跳過已確認不存在的目標
#             if not vehicle.is_alive:
#                 print(f"Vehicle {vehicle_id} destroyed, skipping.")
#                 destroyed_vehicles.add(vehicle_id)  # 加入已銷毀集合
#                 continue
#             try:
#                 location = vehicle.get_location()
#                 velocity = vehicle.get_velocity()
#                 rotation = vehicle.get_transform().rotation
#                 with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
#                     f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
#                             f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
#                             f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
#             except RuntimeError as e:
#                 print(f"Error recording vehicle {vehicle_id}: {e}")
#                 destroyed_vehicles.add(vehicle_id)  # 加入已銷毀集合

#         # Record walker data
#         for walker, walker_controller, walker_id in walkers:
#             if walker_id in destroyed_walkers:
#                 continue  # 跳過已確認不存在的目標
#             if not walker.is_alive:
#                 print(f"Walker {walker_id} destroyed, skipping.")
#                 destroyed_walkers.add(walker_id)  # 加入已銷毀集合
#                 continue
#             try:
#                 location = walker.get_location()
#                 with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
#                     f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
#             except RuntimeError as e:
#                 print(f"Error recording walker {walker_id}: {e}")
#                 destroyed_walkers.add(walker_id)  # 加入已銷毀集合

# import carla
# import os
# import time
# import random
# import cv2
# import numpy as np

# def generate_simulation_data(world, experiment_id, dirname, duration=10, walker_batch_size=100, vehicle_batch_size=50):
#     """
#     Generate vehicles and walkers, record their data, and capture images with feature points.
#     """
#     vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
#     walker_bp_library = world.get_blueprint_library().filter('*walker*')
#     spawn_points = world.get_map().get_spawn_points()

#     # Create directories
#     vehicle_dir = os.path.join(dirname, "vehicle")
#     walker_dir = os.path.join(dirname, "walkers")
#     os.makedirs(vehicle_dir, exist_ok=True)
#     os.makedirs(walker_dir, exist_ok=True)

#     for batch_id in range(0, max(1000, 1000), max(vehicle_batch_size, walker_batch_size)):
#         # Generate vehicles and walkers
#         vehicles, walkers = [], []

#         # Generate vehicles
#         for j in range(batch_id, min(batch_id + vehicle_batch_size, 1000)):
#             blueprint = random.choice(vehicle_bp_library)
#             spawn_point = random.choice(spawn_points)
#             vehicle = world.try_spawn_actor(blueprint, spawn_point)
#             if vehicle:
#                 vehicle.set_autopilot(True)
#                 vehicle_id = f"vehicle_{j}"
#                 vehicles.append((vehicle, vehicle_id))
#                 os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
#                 print(f"Vehicle {j} is generated in experiment {experiment_id}!")

#         # Generate walkers
#         for j in range(batch_id, min(batch_id + walker_batch_size, 1000)):
#             walker_bp = random.choice(walker_bp_library)
#             spawn_point = carla.Transform(location=world.get_random_location_from_navigation())
#             walker = world.try_spawn_actor(walker_bp, spawn_point)
#             if walker:
#                 try:
#                     walker_control_bp = world.get_blueprint_library().find('controller.ai.walker')
#                     walker_controller = world.spawn_actor(walker_control_bp, carla.Transform(), walker)
#                     walker_speed = random.uniform(1.0, 3.0)
#                     walker_controller.start()
#                     walker_controller.set_max_speed(walker_speed)
#                     walker_controller.go_to_location(world.get_random_location_from_navigation())
#                     walker_id = f"walker_{j}"
#                     walkers.append((walker, walker_controller, walker_id))
#                     os.makedirs(f"{walker_dir}/{walker_id}", exist_ok=True)
#                     print(f"Walker {j} is generated with speed {walker_speed:.2f} m/s!")
#                 except Exception as e:
#                     print(f"Failed to initialize walker {j}: {e}")
#                     if walker.is_alive:
#                         walker.destroy()

#         # Simulate and record
#         simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration)

#         # Clean up vehicles
#         for vehicle, _ in vehicles:
#             if vehicle.is_alive:
#                 vehicle.destroy()

#         # Clean up walkers
#         for walker, walker_controller, _ in walkers:
#             if walker_controller.is_alive:
#                 walker_controller.stop()
#                 walker_controller.destroy()
#             if walker.is_alive:
#                 walker.destroy()

#         print(f"Batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1} completed.")

#     print("All simulation batches completed!")



# def simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration):
#     """
#     Simulate vehicles and walkers, record their movements, and capture images.
#     """
#     for t in range(0, duration):
#         world.tick()
#         time.sleep(1)

#         # Record vehicle data
#         for vehicle, vehicle_id in vehicles:
#             if not vehicle.is_alive:
#                 print(f"Vehicle {vehicle_id} is destroyed, skipping.")
#                 continue
#             try:
#                 location = vehicle.get_location()
#                 velocity = vehicle.get_velocity()
#                 rotation = vehicle.get_transform().rotation
#                 with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
#                     f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
#                             f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
#                             f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
#                 capture_single_vehicle_image(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t)
#             except RuntimeError as e:
#                 print(f"Error operating on vehicle {vehicle_id}: {e}")

#         # Record walker data
#         for walker, walker_controller, walker_id in walkers:
#             if not walker.is_alive:
#                 print(f"Walker {walker_id} is no longer alive.")
#                 continue
#             try:
#                 location = walker.get_location()
#                 with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
#                     f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
#             except RuntimeError as e:
#                 print(f"Error operating on walker {walker_id}: {e}")

def capture_single_vehicle_image(world, vehicle, dirname, timestamp):
    """
    Capture a single image from a vehicle's camera and save it with the timestamp.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

    def save_image(image):
        image.save_to_disk(f"{dirname}/{timestamp}.png")

    camera.listen(save_image)
    time.sleep(1)
    camera.stop()
    camera.destroy()

# def generate_simulation_data(world, experiment_id, dirname, duration=10, walker_batch_size=100, vehicle_batch_size=50):
#     """
#     Generate vehicles and walkers, record their data, and capture images with feature points.
#     """
#     vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
#     walker_bp_library = world.get_blueprint_library().filter('*walker*')
#     spawn_points = world.get_map().get_spawn_points()

#     # Create directories
#     vehicle_dir = os.path.join(dirname, "vehicle")
#     walker_dir = os.path.join(dirname, "walkers")
#     os.makedirs(vehicle_dir, exist_ok=True)
#     os.makedirs(walker_dir, exist_ok=True)

#     # Vehicle and walker generation in batches
#     for batch_id in range(0, max(1000, 1000), max(vehicle_batch_size, walker_batch_size)):
#         vehicles = []
#         walkers = []

#         # Generate vehicles
#         for j in range(batch_id, min(batch_id + vehicle_batch_size, 1000)):
#             blueprint = random.choice(vehicle_bp_library)
#             spawn_point = random.choice(spawn_points)
#             vehicle = world.try_spawn_actor(blueprint, spawn_point)
#             if vehicle:
#                 vehicle.set_autopilot(True)
#                 vehicle_id = f"vehicle_{j}"
#                 vehicles.append((vehicle, vehicle_id))
#                 os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
#                 print(f"Vehicle {j} is generated in experiment {experiment_id}!")

#         # Generate walkers
#         for j in range(batch_id, min(batch_id + walker_batch_size, 1000)):
#             walker_bp = random.choice(walker_bp_library)
#             spawn_point = carla.Transform(location=world.get_random_location_from_navigation())
#             walker = world.try_spawn_actor(walker_bp, spawn_point)

#             if walker:
#                 try:
#                     walker_control_bp = world.get_blueprint_library().find('controller.ai.walker')
#                     walker_controller = world.spawn_actor(walker_control_bp, carla.Transform(), walker)
#                     walker_speed = random.uniform(1.0, 3.0)
#                     walker_controller.start()
#                     walker_controller.set_max_speed(walker_speed)
#                     walker_controller.go_to_location(world.get_random_location_from_navigation())
#                     walker.set_simulate_physics(False)

#                     walker_id = f"walker_{j}"
#                     walkers.append((walker, walker_controller, walker_id))
#                     os.makedirs(f"{walker_dir}/{walker_id}", exist_ok=True)
#                     print(f"Walker {j} is generated with speed {walker_speed:.2f} m/s!")
#                 except Exception as e:
#                     print(f"Failed to initialize walker {j}: {e}")
#                     if walker.is_alive:
#                         walker.destroy()

#         # Simulate and record vehicle and walker data
#         simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration)

#         # Cleanup
#         for vehicle, _ in vehicles:
#             if vehicle.is_alive:
#                 vehicle.destroy()

#         for walker, walker_controller, _ in walkers:
#             if walker_controller.is_alive:
#                 walker_controller.stop()
#                 walker_controller.destroy()
#             if walker.is_alive:
#                 walker.destroy()

#         print(f"Batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1} completed.")

#     print("All simulation batches completed!")
# def simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration):
#     """
#     Simulate vehicles and walkers, record their movements, and capture images.
#     """
#     for t in range(0, duration):
#         world.tick()
#         time.sleep(1)

#         # Record vehicle data
#         for vehicle, vehicle_id in vehicles:
#             if not vehicle.is_alive:
#                 print(f"Vehicle {vehicle_id} is destroyed, skipping.")
#                 continue
#             location = vehicle.get_location()
#             velocity = vehicle.get_velocity()
#             rotation = vehicle.get_transform().rotation
#             with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
#                 f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
#                         f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
#                         f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
#             capture_single_vehicle_image(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t)

#         # Record walker data
#         alive_walkers = []
#         for walker, walker_controller, walker_id in walkers:
#             if not walker.is_alive:
#                 print(f"Walker {walker_id} is no longer alive. Cleaning up.")
#                 if walker_controller.is_alive:
#                     walker_controller.stop()
#                     walker_controller.destroy()
#                 continue
#             location = walker.get_location()
#             with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
#                 f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
#             alive_walkers.append((walker, walker_controller, walker_id))
#         walkers[:] = alive_walkers


import carla
import os
import time
import random

def generate_simulation_data(world, experiment_id, dirname, duration=10, walker_batch_size=100, vehicle_batch_size=50):
    """
    Generate vehicles and walkers, record their data, and capture images with feature points.
    """
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    walker_bp_library = world.get_blueprint_library().filter('*walker*')
    spawn_points = world.get_map().get_spawn_points()

    # Create directories
    vehicle_dir = os.path.join(dirname, "vehicle")
    walker_dir = os.path.join(dirname, "walkers")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(walker_dir, exist_ok=True)

    # 記錄不存在的目標
    destroyed_vehicles = set()
    destroyed_walkers = set()

    for batch_id in range(0, 1000, max(vehicle_batch_size, walker_batch_size)):
        vehicles, walkers = [], []

        # Generate vehicles
        for j in range(batch_id, min(batch_id + vehicle_batch_size, 1000)):
            if f"vehicle_{j}" in destroyed_vehicles:
                continue  # 跳過已確認不存在的目標
            spawn_point = random.choice(spawn_points)
            blueprint = random.choice(vehicle_bp_library)
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicle_id = f"vehicle_{j}"
                vehicles.append((vehicle, vehicle_id))
                os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                print(f"Vehicle {j} generated in experiment {experiment_id}.")

        # Generate walkers
        for j in range(batch_id, min(batch_id + walker_batch_size, 1000)):
            if f"walker_{j}" in destroyed_walkers:
                continue  # 跳過已確認不存在的目標
            spawn_point = world.get_random_location_from_navigation()
            if not spawn_point:
                print(f"Invalid spawn point for walker {j}, skipping.")
                continue

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
                    print(f"Walker {j} generated with speed {walker_speed:.2f} m/s.")
                except RuntimeError as e:
                    print(f"Failed to initialize walker {j}: {e}")
                    if walker.is_alive:
                        walker.destroy()

        # Simulate and record
        simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers)

        # Clean up vehicles
        for vehicle, vehicle_id in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        # Clean up walkers
        for walker, walker_controller, walker_id in walkers:
            if walker_controller.is_alive:
                walker_controller.stop()
                walker_controller.destroy()
            if walker.is_alive:
                walker.destroy()

        print(f"Batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1} completed.")

    print("All simulation batches completed!")

def simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers):
    """
    Simulate vehicles and walkers, record their movements, and capture images.
    """
    for t in range(duration):
        world.tick()
        time.sleep(1)

        # Record vehicle data
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue  # Skip destroyed vehicles
            try:
                if not vehicle.is_alive:
                    print(f"Vehicle {vehicle_id} destroyed, skipping.")
                    destroyed_vehicles.add(vehicle_id)
                    continue
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                capture_single_vehicle_image(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t)
            except RuntimeError as e:
                print(f"Error operating on vehicle {vehicle_id}: {e}")
                destroyed_vehicles.add(vehicle_id)

        # Record walker data
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue  # Skip destroyed walkers
            try:
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
            except RuntimeError as e:
                print(f"Error operating on walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

    print(f"Time {t}: Simulation step completed.")

def capture_single_vehicle_image(world, vehicle, dirname, timestamp):
    """
    Capture a single image from a vehicle's camera and save it with the timestamp.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

    def save_image(image):
        image.save_to_disk(f"{dirname}/{timestamp}.png")

    camera.listen(save_image)
    time.sleep(1)
    camera.stop()
    camera.destroy()

import carla
import os
import time
import random

def generate_simulation_data(world, experiment_id, dirname, duration=10, walker_batch_size=10, vehicle_batch_size=5):
    """
    Generate vehicles and walkers, record their data, and capture images with feature points.
    """
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

    for batch_id in range(0, 100, max(vehicle_batch_size, walker_batch_size)):
        vehicles, walkers = [], []

        # Generate vehicles
        for j in range(batch_id, min(batch_id + vehicle_batch_size, 100)):
            if f"vehicle_{j}" in destroyed_vehicles:
                continue

            for attempt in range(3):  # 重試最多 3 次
                spawn_point = random.choice(spawn_points)
                blueprint = random.choice(vehicle_bp_library)
                vehicle = world.try_spawn_actor(blueprint, spawn_point)
                if vehicle:
                    vehicle.set_autopilot(True)
                    vehicle_id = f"vehicle_{j}"
                    vehicles.append((vehicle, vehicle_id))
                    os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                    print(f"Vehicle {j} generated in experiment {experiment_id}.")
                    break
                else:
                    time.sleep(2)  # 等待一段時間後再嘗試

        # Generate walkers
        for j in range(batch_id, min(batch_id + walker_batch_size, 1000)):
            if f"walker_{j}" in destroyed_walkers:
                continue

            for attempt in range(3):  # 重試最多 3 次
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
                        print(f"Walker {j} generated with speed {walker_speed:.2f} m/s.")
                        break
                    except RuntimeError as e:
                        print(f"Failed to initialize walker {j}: {e}")
                        if walker.is_alive:
                            walker.destroy()

        # Simulate and record
        simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers, experiment_id)

        # Clean up vehicles
        for vehicle, vehicle_id in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        # Clean up walkers
        for walker, walker_controller, walker_id in walkers:
            if walker_controller.is_alive:
                walker_controller.stop()
                walker_controller.destroy()
            if walker.is_alive:
                walker.destroy()

        print(f"Batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1} completed.")

    print("All simulation batches completed!")

def simulate_and_record(world, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers, experiment_id):
    """
    Simulate vehicles and walkers, record their movements, and capture images.
    """
    for t in range(duration):
        world.tick()
        time.sleep(1)

        # Record vehicle data
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue
            try:
                if not vehicle.is_alive:
                    print(f"Vehicle {vehicle_id} destroyed, skipping.")
                    destroyed_vehicles.add(vehicle_id)
                    continue

                # 檢查 Z 軸是否偏移並校正
                location = vehicle.get_location()
                if location.z > 0.5 or location.z < -0.5:
                    print(f"Vehicle {vehicle_id} off ground. Correcting position...")
                    location.z = 0.0
                    vehicle.set_transform(carla.Transform(location, vehicle.get_transform().rotation))

                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                    print(f"vehicle_{vehicle_id} record complete at time {t} in experiment {experiment_id}.")
                capture_single_vehicle_image(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t)
                print(f"vehicle_{vehicle_id} image capture complete at time {t} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error operating on vehicle {vehicle_id}: {e}")
                destroyed_vehicles.add(vehicle_id)

        # Record walker data
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t}: Location: ({location.x}, {location.y}, {location.z})\n")
                    print(f"walker_{walker_id} record complete at time {t} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error operating on walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

    print(f"Time {t}: Simulation step completed.")

def capture_single_vehicle_image(world, vehicle, dirname, timestamp):
    """
    Capture a single image from a vehicle's camera and save it with the timestamp.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

    def save_image(image):
        image.save_to_disk(f"{dirname}/{timestamp}.png")

    camera.listen(save_image)
    time.sleep(1)
    camera.stop()
    camera.destroy()



import carla
import os
import time
import random
import cv2
import numpy as np


def generate_simulation_data(world, experiment_id, dirname, duration=20, interval=2, walker_batch_size=10, vehicle_batch_size=5):
    """
    Generate vehicles and walkers, record their data, and capture images with feature points.
    """
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

    for batch_id in range(0, 100, max(vehicle_batch_size, walker_batch_size)):  # Adjusted range for smaller batches
        vehicles, walkers = [], []

        print(f"Starting batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1}...")

        # Generate vehicles
        for j in range(batch_id, min(batch_id + vehicle_batch_size, 100)):
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

        # Generate walkers
        for j in range(batch_id, min(batch_id + walker_batch_size, 1000)):
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

        print(f"Batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1} generation complete. Starting simulation...")

        simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, interval, destroyed_vehicles, destroyed_walkers)

        # Clean up vehicles
        for vehicle, vehicle_id in vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
                    print(f"Vehicle {vehicle_id} successfully cleaned up.")
            except RuntimeError:
                print(f"Error cleaning up vehicle {vehicle_id}.")

        # Clean up walkers
        for walker, walker_controller, walker_id in walkers:
            try:
                if walker_controller.is_alive:
                    walker_controller.stop()
                    walker_controller.destroy()
                    print(f"Walker controller for {walker_id} successfully cleaned up.")
                if walker.is_alive:
                    walker.destroy()
                    print(f"Walker {walker_id} successfully cleaned up.")
            except RuntimeError:
                print(f"Error cleaning up walker {walker_id}.")

        print(f"Batch {batch_id // max(vehicle_batch_size, walker_batch_size) + 1} completed.")

    print("All simulation batches completed!")


def simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, interval, destroyed_vehicles, destroyed_walkers):
    """
    Simulate vehicles and walkers, record their movements, and capture images.
    """
    steps = duration // interval
    for t in range(steps):
        world.tick()
        time.sleep(interval)

        # Record vehicle data
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue
            try:
                if not vehicle.is_alive:
                    print(f"Vehicle {vehicle_id} destroyed, skipping.")
                    destroyed_vehicles.add(vehicle_id)
                    continue

                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                print(f"Vehicle {vehicle_id} data recorded at time {t * interval} in experiment {experiment_id}.")
                capture_single_vehicle_image(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t * interval)
                print(f"Vehicle {vehicle_id} image captured at time {t * interval} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error operating on vehicle {vehicle_id}: {e}")
                destroyed_vehicles.add(vehicle_id)

        # Record walker data
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z})\n")
                print(f"Walker {walker_id} data recorded at time {t * interval} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error operating on walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

    print(f"Simulation for time step {t * interval} completed.")


def capture_single_vehicle_image(world, vehicle, dirname, timestamp):
    """
    Capture a single image from a vehicle's camera and save it with the timestamp.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

    def save_image(image):
        image.save_to_disk(f"{dirname}/{timestamp}.png")

    camera.listen(save_image)
    time.sleep(1)
    camera.stop()
    camera.destroy()


import carla
import os
import time
import random
import cv2
import numpy as np

def clear_existing_actors(client):
    """
    清除所有已存在的車輛和行人。
    """
    world = client.get_world()
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
def generate_simulation_data(client, experiment_id, dirname, duration=10, interval=1):
    """
    Generate vehicles and walkers, record their data, and capture images with feature points.
    """
    world = client.get_world()
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
    clear_existing_actors(client)

    # Generate all vehicles
    print("Starting vehicle generation...")
    for j in range(100):  # Total 100 vehicles
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

    # Generate all walkers
    print("Starting walker generation...")
    for j in range(200):  # Total 200 walkers
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

    # Start simulation and recording
    print("Starting simulation...")
    simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, interval, destroyed_vehicles, destroyed_walkers)

    # Clean up vehicles
    print("Cleaning up vehicles...")
    for vehicle, vehicle_id in vehicles:
        try:
            if vehicle.is_alive:
                vehicle.destroy()
                print(f"Vehicle {vehicle_id} successfully cleaned up.")
        except RuntimeError:
            print(f"Error cleaning up vehicle {vehicle_id}.")

    # Clean up walkers
    print("Cleaning up walkers...")
    for walker, walker_controller, walker_id in walkers:
        try:
            if walker_controller.is_alive:
                walker_controller.stop()
                walker_controller.destroy()
                print(f"Walker controller for {walker_id} successfully cleaned up.")
            if walker.is_alive:
                walker.destroy()
                print(f"Walker {walker_id} successfully cleaned up.")
        except RuntimeError:
            print(f"Error cleaning up walker {walker_id}.")

    print("Simulation completed!")

def simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, interval, destroyed_vehicles, destroyed_walkers):
    """
    Simulate vehicles and walkers, record their movements, and capture images.
    """
    steps = duration // interval
    for t in range(steps):
        world.tick()
        time.sleep(interval)

        # Record vehicle data
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue
            try:
                # 確保車輛仍然存活
                if not vehicle.is_alive:
                    print(f"Vehicle {vehicle_id} destroyed, skipping.")
                    destroyed_vehicles.add(vehicle_id)
                    continue

                # 紀錄車輛位置和速度數據
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                print(f"Vehicle {vehicle_id} data recorded at time {t * interval} in experiment {experiment_id}.")

                # 捕獲圖像並檢測特徵點
                capture_images_with_feature_points(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t * interval)
                print(f"Vehicle {vehicle_id} image captured at time {t * interval} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error operating on vehicle {vehicle_id}: {e}")
                destroyed_vehicles.add(vehicle_id)

        # Record walker data
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                # 確保行人仍然存活
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue

                # 紀錄行人位置數據
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z})\n")
                print(f"Walker {walker_id} data recorded at time {t * interval} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error operating on walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

    print("Simulation completed for time step.")


def capture_images_with_feature_points(world, vehicle, dirname, timestamp):
    """
    Capture RGB and depth images from a vehicle's camera, detect feature points,
    and save 2D and 3D feature points to files.
    """
    # Create RGB camera
    rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    rgb_camera_bp.set_attribute("image_size_x", "1280")
    rgb_camera_bp.set_attribute("image_size_y", "720")
    rgb_camera_bp.set_attribute("fov", "90")
    rgb_transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
    rgb_camera = world.spawn_actor(rgb_camera_bp, rgb_transform, attach_to=vehicle)

    # Create depth camera
    depth_camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
    depth_camera_bp.set_attribute("image_size_x", "1280")
    depth_camera_bp.set_attribute("image_size_y", "720")
    depth_camera_bp.set_attribute("fov", "90")
    depth_camera = world.spawn_actor(depth_camera_bp, rgb_transform, attach_to=vehicle)

    # Containers for captured data
    rgb_image_data = []
    depth_image_data = []

    def save_rgb_image(image):
        rgb_image_data.append(image)

    def save_depth_image(image):
        depth_image_data.append(image)

    rgb_camera.listen(save_rgb_image)
    depth_camera.listen(save_depth_image)
    time.sleep(1)  # Wait for the images to be captured
    rgb_camera.stop()
    depth_camera.stop()
    rgb_camera.destroy()
    depth_camera.destroy()

    if rgb_image_data and depth_image_data:
        # Process RGB image
        rgb_image = rgb_image_data[0]
        rgb_array = np.array(rgb_image.raw_data).reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

        # Process depth image
        depth_image = depth_image_data[0]
        depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
        depth_array = depth_array * 1000.0 / 255.0  # Convert to meters

        # Detect feature points in RGB image
        feature_points_2d = detect_feature_points(rgb_array)

        # Map 2D feature points to 3D world coordinates
        feature_points_3d = []
        intrinsic = get_camera_intrinsic(rgb_camera)
        transform = rgb_camera.get_transform()

        for x, y in feature_points_2d:
            depth = depth_array[int(y), int(x)]
            pixel_coords = np.array([x, y, 1.0])
            camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
            world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
            feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])

        # Save 2D and 3D feature points
        with open(f"{dirname}/{timestamp}_2D_FP.txt", "w") as f:
            for x, y in feature_points_2d:
                f.write(f"{x} {y}\n")

        with open(f"{dirname}/{timestamp}_3D_FP.txt", "w") as f:
            for x, y, z in feature_points_3d:
                f.write(f"{x} {y} {z}\n")

        print(f"Feature points saved for vehicle {dirname.split('/')[-1]} at time {timestamp}.")


def detect_feature_points(image):
    """
    Detect feature points in the image using ORB (Oriented FAST and Rotated BRIEF).
    """
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(gray, None)
    return [kp.pt for kp in keypoints]


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


# import carla
# import os
# import time
# import random
# import cv2
# import numpy as np


# def generate_simulation_data(world, experiment_id, dirname, duration=10, interval=1):
#     """
#     Generate vehicles and walkers, record their data, and capture images with feature points.
#     """
#     vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
#     walker_bp_library = world.get_blueprint_library().filter('*walker*')
#     spawn_points = world.get_map().get_spawn_points()

#     # Create directories
#     vehicle_dir = os.path.join(dirname, "vehicle")
#     walker_dir = os.path.join(dirname, "walkers")
#     os.makedirs(vehicle_dir, exist_ok=True)
#     os.makedirs(walker_dir, exist_ok=True)

#     destroyed_vehicles = set()
#     destroyed_walkers = set()

#     vehicles, walkers = [], []

#     # Generate all vehicles
#     print("Starting vehicle generation...")
#     for j in range(100):  # Total 100 vehicles
#         if f"vehicle_{j}" in destroyed_vehicles:
#             continue
#         for attempt in range(3):  # Retry up to 3 times
#             spawn_point = random.choice(spawn_points)
#             blueprint = random.choice(vehicle_bp_library)
#             vehicle = world.try_spawn_actor(blueprint, spawn_point)
#             if vehicle:
#                 vehicle.set_autopilot(True)
#                 vehicle_id = f"vehicle_{j}"
#                 vehicles.append((vehicle, vehicle_id))
#                 os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
#                 print(f"Vehicle {vehicle_id} successfully generated in experiment {experiment_id}.")
#                 break
#             else:
#                 print(f"Attempt {attempt + 1} failed to spawn vehicle {j}. Retrying...")
#                 time.sleep(2)

#     print(f"Completed vehicle generation: {len(vehicles)} vehicles generated.")

#     # Generate all walkers
#     print("Starting walker generation...")
#     for j in range(200):  # Total 200 walkers
#         if f"walker_{j}" in destroyed_walkers:
#             continue
#         for attempt in range(3):
#             spawn_point = world.get_random_location_from_navigation()
#             if not spawn_point:
#                 print(f"Invalid spawn point for walker {j}, skipping.")
#                 break
#             walker_bp = random.choice(walker_bp_library)
#             walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_point))
#             if walker:
#                 try:
#                     walker_control_bp = world.get_blueprint_library().find('controller.ai.walker')
#                     walker_controller = world.spawn_actor(walker_control_bp, carla.Transform(), walker)
#                     walker_speed = random.uniform(1.0, 3.0)
#                     walker_controller.start()
#                     walker_controller.set_max_speed(walker_speed)
#                     walker_controller.go_to_location(world.get_random_location_from_navigation())
#                     walker_id = f"walker_{j}"
#                     walkers.append((walker, walker_controller, walker_id))
#                     os.makedirs(f"{walker_dir}/{walker_id}", exist_ok=True)
#                     print(f"Walker {walker_id} generated with speed {walker_speed:.2f} m/s.")
#                     break
#                 except RuntimeError as e:
#                     print(f"Failed to initialize walker {walker_id}: {e}")
#                     if walker.is_alive:
#                         walker.destroy()

#     print(f"Completed walker generation: {len(walkers)} walkers generated.")

#     # Start simulation and recording
#     print("Starting simulation...")
#     simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, interval, destroyed_vehicles, destroyed_walkers)

#     # Clean up vehicles
#     print("Cleaning up vehicles...")
#     for vehicle, vehicle_id in vehicles:
#         try:
#             if vehicle.is_alive:
#                 vehicle.destroy()
#                 print(f"Vehicle {vehicle_id} successfully cleaned up.")
#         except RuntimeError:
#             print(f"Error cleaning up vehicle {vehicle_id}.")

#     # Clean up walkers
#     print("Cleaning up walkers...")
#     for walker, walker_controller, walker_id in walkers:
#         try:
#             if walker_controller.is_alive:
#                 walker_controller.stop()
#                 walker_controller.destroy()
#                 print(f"Walker controller for {walker_id} successfully cleaned up.")
#             if walker.is_alive:
#                 walker.destroy()
#                 print(f"Walker {walker_id} successfully cleaned up.")
#         except RuntimeError:
#             print(f"Error cleaning up walker {walker_id}.")

#     print("Simulation completed!")


# def simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, interval, destroyed_vehicles, destroyed_walkers):
#     """
#     Simulate vehicles and walkers, record their movements, and capture images.
#     """
#     steps = duration // interval
#     for t in range(steps):
#         world.tick()
#         time.sleep(interval)

#         # Record vehicle data
#         for vehicle, vehicle_id in vehicles:
#             if vehicle_id in destroyed_vehicles:
#                 continue
#             try:
#                 if not vehicle.is_alive:
#                     print(f"Vehicle {vehicle_id} destroyed, skipping.")
#                     destroyed_vehicles.add(vehicle_id)
#                     continue

#                 location = vehicle.get_location()
#                 velocity = vehicle.get_velocity()
#                 rotation = vehicle.get_transform().rotation
#                 with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
#                     f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z}), "
#                             f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
#                             f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
#                 print(f"Vehicle {vehicle_id} data recorded at time {t * interval} in experiment {experiment_id}.")
#                 capture_single_vehicle_image(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t * interval)
#                 print(f"Vehicle {vehicle_id} image captured at time {t * interval} in experiment {experiment_id}.")
#             except RuntimeError as e:
#                 print(f"Error operating on vehicle {vehicle_id}: {e}")
#                 destroyed_vehicles.add(vehicle_id)

#         # Record walker data
#         for walker, walker_controller, walker_id in walkers:
#             if walker_id in destroyed_walkers:
#                 continue
#             try:
#                 if not walker.is_alive:
#                     print(f"Walker {walker_id} destroyed, skipping.")
#                     destroyed_walkers.add(walker_id)
#                     continue
#                 location = walker.get_location()
#                 with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
#                     f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z})\n")
#                 print(f"Walker {walker_id} data recorded at time {t * interval} in experiment {experiment_id}.")
#             except RuntimeError as e:
#                 print(f"Error operating on walker {walker_id}: {e}")
#                 destroyed_walkers.add(walker_id)

#     print("Simulation completed for time step.")

# def capture_single_vehicle_image(world, vehicle, dirname, timestamp):
#     """
#     Capture a single image from a vehicle's camera, detect feature points,
#     and save both the image and feature points' coordinates.
#     """
#     camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
#     camera_bp.set_attribute("image_size_x", "1280")
#     camera_bp.set_attribute("image_size_y", "720")
#     camera_bp.set_attribute("fov", "90")
#     transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
#     camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

#     # Container for captured image
#     image_data = []

#     def save_image(image):
#         image_data.append(image)
#         image.save_to_disk(f"{dirname}/{timestamp}.png")
#         print(f"Image {timestamp}.png saved for vehicle {dirname.split('/')[-1]}.")

#     camera.listen(save_image)
#     time.sleep(1)  # Wait for the image to be captured
#     camera.stop()
#     camera.destroy()

#     if image_data:
#         # Convert CARLA raw image to numpy array
#         image = image_data[0]
#         image_array = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]

#         # Detect feature points using OpenCV ORB
#         feature_points = detect_feature_points(image_array)

#         # Save feature points to a file
#         fp_filename = f"{dirname}/{timestamp}_FP.txt"
#         with open(fp_filename, "w") as f:
#             for x, y in feature_points:
#                 f.write(f"{x} {y}\n")
#         print(f"Feature points saved to {fp_filename}.")

# def detect_feature_points(image):
#     """
#     Detect feature points in the image using ORB (Oriented FAST and Rotated BRIEF).
#     """
#     orb = cv2.ORB_create()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     keypoints = orb.detect(gray, None)
#     feature_points = [kp.pt for kp in keypoints]  # Extract (x, y) coordinates
#     return feature_points


# def get_camera_intrinsic(camera):
#     """
#     Retrieve the intrinsic matrix of the camera.

#     Args:
#         camera: The camera actor to extract intrinsics from.

#     Returns:
#         A 3x3 numpy array representing the intrinsic matrix.
#     """
#     print("--------get_camera_intrinsic------------")
#     fov = float(camera.attributes['fov'])
#     width = int(camera.attributes['image_size_x'])
#     height = int(camera.attributes['image_size_y'])
#     focal_length = width / (2.0 * np.tan(np.radians(fov / 2.0)))
#     cx = width / 2.0
#     cy = height / 2.0
#     print(np.array([
#         [focal_length, 0, cx],
#         [0, focal_length, cy],
#         [0, 0, 1]
#     ]))
#     return np.array([
#         [focal_length, 0, cx],
#         [0, focal_length, cy],
#         [0, 0, 1]
#     ])


# def capture_images_with_feature_points(world, vehicle, dirname, timestamp):
#     """
#     Capture RGB and depth images from a vehicle's camera, detect feature points,
#     and save 2D and 3D feature points to files.
#     """
#     try:
#         # Create RGB camera
#         rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
#         rgb_camera_bp.set_attribute("image_size_x", "1280")
#         rgb_camera_bp.set_attribute("image_size_y", "720")
#         rgb_camera_bp.set_attribute("fov", "90")
#         rgb_transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
#         rgb_camera = world.spawn_actor(rgb_camera_bp, rgb_transform, attach_to=vehicle)

#         # Create depth camera
#         depth_camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
#         depth_camera_bp.set_attribute("image_size_x", "1280")
#         depth_camera_bp.set_attribute("image_size_y", "720")
#         depth_camera_bp.set_attribute("fov", "90")
#         depth_camera = world.spawn_actor(depth_camera_bp, rgb_transform, attach_to=vehicle)

#         # Containers for captured data
#         rgb_image_data = []
#         depth_image_data = []

#         def save_rgb_image(image):
#             rgb_image_data.append(image)

#         def save_depth_image(image):
#             depth_image_data.append(image)

#         rgb_camera.listen(save_rgb_image)
#         depth_camera.listen(save_depth_image)
#         time.sleep(1)  # Wait for the images to be captured
#         rgb_camera.stop()
#         depth_camera.stop()
#         rgb_camera.destroy()
#         depth_camera.destroy()

#         if rgb_image_data and depth_image_data:
#             # Process RGB image
#             rgb_image = rgb_image_data[0]
#             rgb_array = np.array(rgb_image.raw_data).reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

#             # Process depth image
#             depth_image = depth_image_data[0]
#             depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
#             depth_array = depth_array * 1000.0 / 255.0  # Convert to meters

#             # Detect feature points in RGB image
#             feature_points_2d = detect_feature_points(rgb_array)

#             # Map 2D feature points to 3D world coordinates
#             feature_points_3d = []
#             intrinsic = get_camera_intrinsic(rgb_camera)
#             transform = rgb_camera.get_transform()

#             for x, y in feature_points_2d:
#                 depth = depth_array[int(y), int(x)]
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#                 world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
#                 feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])

#             # Save 2D and 3D feature points
#             with open(f"{dirname}/{timestamp}_2D_FP.txt", "w") as f:
#                 for x, y in feature_points_2d:
#                     f.write(f"{x} {y}\n")

#             with open(f"{dirname}/{timestamp}_3D_FP.txt", "w") as f:
#                 for x, y, z in feature_points_3d:
#                     f.write(f"{x} {y} {z}\n")

#             print(f"Feature points saved for vehicle {dirname.split('/')[-1]} at time {timestamp}.")

#     except RuntimeError as e:
#         print(f"Error capturing images or feature points: {e}")

# def detect_feature_points(image):
#     """
#     Detect feature points in the image using ORB (Oriented FAST and Rotated BRIEF).
#     """
#     orb = cv2.ORB_create()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     keypoints = orb.detect(gray, None)
#     return [kp.pt for kp in keypoints]

# def get_camera_intrinsic(camera):
#     """
#     Retrieve the intrinsic matrix of the camera.
#     """
#     fov = float(camera.attributes['fov'])
#     width = int(camera.attributes['image_size_x'])
#     height = int(camera.attributes['image_size_y'])
#     focal_length = width / (2.0 * np.tan(np.radians(fov / 2.0)))
#     cx = width / 2.0
#     cy = height / 2.0

#     return np.array([
#         [focal_length, 0, cx],
#         [0, focal_length, cy],
#         [0, 0, 1]
#     ])
def attach_depth_camera_to_vehicle(world, vehicle):
    print("------------attach_depth_camera_to_vehicle start-----------")
    """
    Attach a depth camera to the vehicle and return the camera actor.
    """
    try:
        depth_camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera_bp.set_attribute("image_size_x", "1280")
        depth_camera_bp.set_attribute("image_size_y", "720")
        depth_camera_bp.set_attribute("fov", "90")
        depth_transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
        depth_camera = world.spawn_actor(depth_camera_bp, depth_transform, attach_to=vehicle)
        return depth_camera
    except RuntimeError as e:
        print(f"Failed to attach depth camera to vehicle: {e}")
        return None

def attach_rgb_camera_to_vehicle(world, vehicle):
    print("------------attach_rgb_camera_to_vehicle start-----------")
    try:
        rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute("image_size_x", "1280")
        rgb_camera_bp.set_attribute("image_size_y", "720")
        rgb_camera_bp.set_attribute("fov", "90")
        rgb_transform = carla.Transform(location=carla.Location(x=2.5, z=1.6))
        rgb_camera = world.spawn_actor(rgb_camera_bp, rgb_transform, attach_to=vehicle)
        return rgb_camera
    except RuntimeError as e:
        print(f"Failed to attach camera to vehicle: {e}")
        return None

# def capture_images_with_feature_points(world, vehicle, dirname, timestamp):
#     """
#     Capture RGB and depth images, ensuring cameras are properly attached and data is saved.
#     """
#     print("------------capture_images_with_feature_points start-----------")
#     try:
#         rgb_camera = attach_rgb_camera_to_vehicle(world, vehicle)
#         if rgb_camera is None:
#             print(f"Skipping image capture for vehicle {dirname.split('/')[-1]} at time {timestamp}: Camera not attached.")
#             return

#         depth_camera = attach_depth_camera_to_vehicle(world, vehicle)
#         if depth_camera is None:
#             print(f"Skipping image capture for vehicle {dirname.split('/')[-1]} at time {timestamp}: Depth camera not attached.")
#             rgb_camera.destroy()
#             return

#         # Containers for image data
#         rgb_image_data = []
#         depth_image_data = []

#         def save_rgb_image(image):
#             rgb_image_data.append(image)

#         def save_depth_image(image):
#             depth_image_data.append(image)
#         print("------266-------------266------------")

#         rgb_camera.listen(save_rgb_image)
#         depth_camera.listen(save_depth_image)
#         time.sleep(1)  # Allow images to be captured
#         # 在銷毀前提取相機參數
#         intrinsic = get_camera_intrinsic(rgb_camera)
#         transform = rgb_camera.get_transform()

#         rgb_camera.stop()
#         depth_camera.stop()
#         rgb_camera.destroy()
#         depth_camera.destroy()
#         print("------275-------------275------------")

#         if rgb_image_data and depth_image_data:
#             # Process and save feature points
#             # process_and_save_feature_points(rgb_image_data[0], depth_image_data[0], dirname, timestamp)
#             print("------280-------------280------------")
#             process_and_save_feature_points(rgb_image_data[0], depth_image_data[0], intrinsic, transform, dirname, timestamp)

#         else:
#             print(f"No images captured for vehicle {dirname.split('/')[-1]} at time {timestamp}.")

#     except RuntimeError as e:
#         print(f"Error during image capture for vehicle {dirname.split('/')[-1]}: {e}")

# def process_and_save_feature_points(rgb_image, depth_image, dirname, timestamp):
#     """
#     Process RGB and depth images to extract 2D and 3D feature points
#     and save them to corresponding files.

#     Args:
#         rgb_image: The RGB image captured from the camera.
#         depth_image: The depth image captured from the camera.
#         dirname: Directory where files will be saved.
#         timestamp: Timestamp for file naming.
#     """
#     # Convert RGB image to numpy array
#     rgb_array = np.array(rgb_image.raw_data).reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

#     # Convert depth image to numpy array and normalize to meters
#     depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
#     depth_array = depth_array * 1000.0 / 255.0  # Convert to meters

#     # Detect 2D feature points in RGB image
#     feature_points_2d = detect_feature_points(rgb_array)

#     # Map 2D feature points to 3D world coordinates
#     feature_points_3d = []
#     intrinsic = get_camera_intrinsic(rgb_image)
#     transform = rgb_image.transform

#     for x, y in feature_points_2d:
#         try:
#             depth = depth_array[int(y), int(x)]
#             if depth <= 0:  # Ignore invalid depth values
#                 continue
#             pixel_coords = np.array([x, y, 1.0])
#             camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#             world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
#             feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])
#         except Exception as e:
#             print(f"Error processing feature point at ({x}, {y}): {e}")

#     # Save 2D feature points
#     with open(f"{dirname}/{timestamp}_2D_FP.txt", "w") as f:
#         for x, y in feature_points_2d:
#             f.write(f"{x} {y}\n")

#     # Save 3D feature points
#     with open(f"{dirname}/{timestamp}_3D_FP.txt", "w") as f:
#         for x, y, z in feature_points_3d:
#             f.write(f"{x} {y} {z}\n")

#     print(f"Feature points saved for timestamp {timestamp} in {dirname}.")
# def process_and_save_feature_points(rgb_image, depth_image, camera, dirname, timestamp):
#     """
#     Process RGB and depth images to extract 2D and 3D feature points
#     and save them to corresponding files.

#     Args:
#         rgb_image: The RGB image captured from the camera.
#         depth_image: The depth image captured from the camera.
#         camera: The camera actor to extract intrinsic parameters.
#         dirname: Directory where files will be saved.
#         timestamp: Timestamp for file naming.
#     """
#     print("-------process_and_save_feature_points start------")
#     # Convert RGB image to numpy array
#     rgb_array = np.array(rgb_image.raw_data).reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]

#     # Convert depth image to numpy array and normalize to meters
#     depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
#     depth_array = depth_array * 1000.0 / 255.0  # Convert to meters

#     # Detect 2D feature points in RGB image
#     feature_points_2d = detect_feature_points(rgb_array)
#     print("--------360-----------360------------")
#     # Map 2D feature points to 3D world coordinates
#     feature_points_3d = []
#     intrinsic = get_camera_intrinsic(camera)
#     print(f"intrinsic: {intrinsic}")
#     transform = camera.get_transform()
#     print(f"transform: {transform}")
#     for x, y in feature_points_2d:
#         try:
#             depth = depth_array[int(y), int(x)]
#             if depth <= 0:  # Ignore invalid depth values
#                 continue
#             pixel_coords = np.array([x, y, 1.0])
#             camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#             world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
#             feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])
#         except Exception as e:
#             print(f"Error processing feature point at ({x}, {y}): {e}")

#     # Save 2D feature points
#     with open(f"{dirname}/{timestamp}_2D_FP.txt", "w") as f:
#         for x, y in feature_points_2d:
#             f.write(f"{x} {y}\n")
#     print("--------382------------")
#     # Save 3D feature points
#     with open(f"{dirname}/{timestamp}_3D_FP.txt", "w") as f:
#         for x, y, z in feature_points_3d:
#             f.write(f"{x} {y} {z}\n")

#     print(f"Feature points saved for timestamp {timestamp} in {dirname}.")

# Simulate and record
def simulate_and_record(experiment_id, world, vehicles, walkers, vehicle_dir, walker_dir, duration, interval, destroyed_vehicles, destroyed_walkers):
    """
    Simulate vehicles and walkers, record their movements, and capture images.
    """
    steps = duration // interval
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    for t in range(steps):

        world.tick()
        time.sleep(2)

        # 檢查車輛是否存活，若有損失則補充
        for i, (vehicle, vehicle_id) in enumerate(vehicles):
            if not vehicle.is_alive:  # 修正此處，移除括號
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

        # 紀錄車輛數據並捕獲圖像
        for vehicle, vehicle_id in vehicles:
            if vehicle_id in destroyed_vehicles:
                continue
            try:
                if not vehicle.is_alive:
                    print(f"Vehicle {vehicle_id} destroyed, skipping.")
                    destroyed_vehicles.add(vehicle_id)
                    continue

                # 紀錄車輛位置和速度數據
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                rotation = vehicle.get_transform().rotation
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z}), "
                            f"Velocity: ({velocity.x}, {velocity.y}, {velocity.z}), "
                            f"Rotation: ({rotation.pitch}, {rotation.yaw}, {rotation.roll})\n")
                print(f"Vehicle {vehicle_id} data recorded at time {t * interval} in experiment {experiment_id}.")
                # 捕獲圖像並檢測特徵點
                capture_images_with_feature_points(world, vehicle, f"{vehicle_dir}/{vehicle_id}", t * interval)
            except RuntimeError as e:
                print(f"Error operating on vehicle {vehicle_id}: {e}")
                destroyed_vehicles.add(vehicle_id)

        # 紀錄行人數據
        for walker, walker_controller, walker_id in walkers:
            if walker_id in destroyed_walkers:
                continue
            try:
                if not walker.is_alive:
                    print(f"Walker {walker_id} destroyed, skipping.")
                    destroyed_walkers.add(walker_id)
                    continue

                # 紀錄行人位置數據
                location = walker.get_location()
                with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
                    f.write(f"Time {t * interval}: Location: ({location.x}, {location.y}, {location.z})\n")
                print(f"Walker {walker_id} data recorded at time {t * interval} in experiment {experiment_id}.")
            except RuntimeError as e:
                print(f"Error operating on walker {walker_id}: {e}")
                destroyed_walkers.add(walker_id)

    print("Simulation completed for time step.")



# def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file):
#     """
#     Capture RGB and depth images, convert 2D feature points to 3D world coordinates,
#     and match them with points in PCD_FP_town01.txt.

#     Args:
#         world: CARLA world object.
#         vehicle: Vehicle actor to attach cameras.
#         dirname: Directory where outputs will be saved.
#         timestamp: Timestamp for file naming.
#         fp_town_file: Path to PCD_FP_town01.txt containing world coordinates.
#     """
#     # Attach cameras
#     rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
#     depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

#     if not rgb_camera or not depth_camera:
#         print(f"Skipping capture for vehicle {dirname.split('/')[-1]} at time {timestamp}: Cameras not attached.")
#         if rgb_camera:
#             rgb_camera.destroy()
#         if depth_camera:
#             depth_camera.destroy()
#         return

#     # Load PCD_FP_town01.txt points
#     try:
#         fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
#     except Exception as e:
#         print(f"Error loading PCD_FP_town01.txt: {e}")
#         return

#     # Containers for captured data
#     rgb_image_data = []
#     depth_image_data = []

#     def save_rgb_image(image):
#         rgb_image_data.append(image)
#         image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")
#         # print(f"RGB image saved at {dirname}/rgb_{timestamp}.png.")

#     def save_depth_image(image):
#         depth_image_data.append(image)
#         depth_array = np.array(image.raw_data, dtype=np.float32).reshape((image.height, image.width, 4))[:, :, 0]
#         depth_array_normalized = (depth_array / np.max(depth_array) * 255.0).astype(np.uint8)
#         depth_image_path = f"{dirname}/depth_{timestamp}.png"
#         cv2.imwrite(depth_image_path, depth_array_normalized)
#         # print(f"Grayscale depth image saved at {depth_image_path}.")

#     rgb_camera.listen(save_rgb_image)
#     depth_camera.listen(save_depth_image)

#     # Wait for images to be captured
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if rgb_image_data and depth_image_data:
#             break

#     # Stop listening to cameras
#     rgb_camera.stop()
#     depth_camera.stop()

#     if not (rgb_image_data and depth_image_data):
#         print(f"Failed to capture images for vehicle {dirname.split('/')[-1]} at time {timestamp}.")
#     else:
#         # Precompute intrinsic and transform
#         intrinsic = get_camera_intrinsic(rgb_camera)
#         transform = rgb_camera.get_transform()

#         # Detect feature points in the RGB image
#         rgb_array = np.array(rgb_image_data[0].raw_data).reshape((720, 1280, 4))[:, :, :3]
#         detected_fp = detect_feature_points(rgb_array)

#         # Convert 2D feature points to 3D world coordinates
#         feature_points_3d = []
#         depth_array = np.array(depth_image_data[0].raw_data, dtype=np.float32).reshape((720, 1280, 4))[:, :, 0]

#         for x, y in detected_fp:
#             try:
#                 depth = depth_array[int(y), int(x)]
#                 if depth <= 0:
#                     continue  # Skip invalid depth points

#                 # Convert to camera coordinates
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth

#                 # Convert to world coordinates
#                 world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
#                 feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])
#             except Exception as e:
#                 print(f"Error converting 2D FP ({x}, {y}) to 3D world coordinates: {e}")

#         feature_points_3d = np.array(feature_points_3d)

#         # Match converted points with PCD_FP_town01.txt
#         match_count = sum(
#             1 for wp in feature_points_3d
#             if any(np.linalg.norm(wp - tp) < 5 for tp in fp_town_points)
#         )

#         if match_count >= 4:
#             print(f"---------Image at {timestamp} satisfies conditions. Proceeding with data processing.---------")

#             # Save grayscale depth image
#             save_depth_image(depth_image_data[0])

#             # Process and save feature points
#             process_and_save_feature_points(rgb_image_data[0], depth_image_data[0], intrinsic, transform, dirname, timestamp)

#             # Save .ply file from depth image
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_ply_file(depth_image_data[0], intrinsic, transform, ply_filename)

#             # Estimate camera pose
#             try:
#                 detected_fp_array = np.array(detected_fp, dtype=np.float32)
#                 fp_town_points_array = np.array(fp_town_points[:len(detected_fp)], dtype=np.float32)
#                 rvec, tvec = estimate_camera_pose_2d_3d(detected_fp_array, fp_town_points_array)

#                 with open(f"{dirname}/pose_{timestamp}.txt", "w") as pose_file:
#                     pose_file.write(f"Rotation Vector: {rvec.tolist()}\n")
#                     pose_file.write(f"Translation Vector: {tvec.tolist()}\n")
#                 print(f"Camera pose saved at {dirname}/pose_{timestamp}.txt.")
#             except Exception as e:
#                 print(f"Error estimating camera pose: {e}")
#         else:
#             print(f"Image at {timestamp} does not satisfy conditions. Skipping.")

#     # Destroy cameras
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()

# def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file):
#     """
#     Capture RGB and depth images from a vehicle's camera, detect feature points,
#     and save 2D/3D feature points and images to files.

#     Args:
#         world: CARLA world object.
#         vehicle: Vehicle actor to attach cameras.
#         dirname: Directory where outputs will be saved.
#         timestamp: Timestamp for file naming.
#         fp_town_file: Path to FP_town01.txt containing world coordinates of interest.
#     """
#     # Attach cameras
#     rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
#     depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

#     if not rgb_camera or not depth_camera:
#         print(f"Skipping capture for vehicle {dirname.split('/')[-1]} at time {timestamp}: Cameras not attached.")
#         if rgb_camera:
#             rgb_camera.destroy()
#         if depth_camera:
#             depth_camera.destroy()
#         return

#     # Load FP_town01.txt points
#     try:
#         fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
#     except Exception as e:
#         print(f"Error loading FP_town01.txt: {e}")
#         return

#     # Containers for captured data
#     rgb_image_data = []
#     depth_image_data = []

#     def save_rgb_image(image):
#         rgb_image_data.append(image)
#         image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")
#         # print(f"RGB image saved at {dirname}/rgb_{timestamp}.png.")

#     def save_depth_image(image):
#         depth_image_data.append(image)
#         depth_array = np.array(image.raw_data, dtype=np.float32).reshape((image.height, image.width, 4))[:, :, 0]
#         depth_array_normalized = (depth_array / np.max(depth_array) * 255.0).astype(np.uint8)
#         depth_image_path = f"{dirname}/depth_{timestamp}.png"
#         cv2.imwrite(depth_image_path, depth_array_normalized)
#         # print(f"Grayscale depth image saved at {depth_image_path}.")

#     rgb_camera.listen(save_rgb_image)
#     depth_camera.listen(save_depth_image)

#     # Wait for images to be captured
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if rgb_image_data and depth_image_data:
#             break

#     # Stop listening to cameras
#     rgb_camera.stop()
#     depth_camera.stop()

#     if not (rgb_image_data and depth_image_data):
#         print(f"Failed to capture images for vehicle {dirname.split('/')[-1]} at time {timestamp}.")
#     else:
#         # Precompute intrinsic and transform
#         intrinsic = get_camera_intrinsic(rgb_camera)
#         transform = rgb_camera.get_transform()

#         # Detect feature points in the RGB image
#         rgb_array = np.array(rgb_image_data[0].raw_data).reshape((720, 1280, 4))[:, :, :3]
#         detected_fp = detect_feature_points(rgb_array)

#         # Check for matches with FP_town01.txt
#         match_count = sum(
#             1 for x, y in detected_fp
#             if any(np.linalg.norm(fp[:2] - [x, y]) < 5 for fp in fp_town_points)
#         )

#         if match_count >= 4:
#             print(f"---------Image at {timestamp} satisfies conditions. Proceeding with data processing.---------")

#             # Save grayscale depth image
#             save_depth_image(depth_image_data[0])

#             # Process and save feature points
#             process_and_save_feature_points(rgb_image_data[0], depth_image_data[0], intrinsic, transform, dirname, timestamp)

#             # Save .ply file from depth image
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_ply_file(depth_image_data[0], intrinsic, transform, ply_filename)

#             # Estimate camera pose
#             try:
#                 detected_fp_array = np.array(detected_fp, dtype=np.float32)
#                 fp_town_points_array = np.array(fp_town_points[:len(detected_fp)], dtype=np.float32)
#                 rvec, tvec = estimate_camera_pose_2d_3d(detected_fp_array, fp_town_points_array)

#                 with open(f"{dirname}/pose_{timestamp}.txt", "w") as pose_file:
#                     pose_file.write(f"Rotation Vector: {rvec.tolist()}\n")
#                     pose_file.write(f"Translation Vector: {tvec.tolist()}\n")
#                 print(f"Camera pose saved at {dirname}/pose_{timestamp}.txt.")
#             except Exception as e:
#                 print(f"Error estimating camera pose: {e}")
#         else:
#             print(f"Image at {timestamp} does not satisfy conditions. Skipping.")

#     # Destroy cameras
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()


# def process_actor_for_fp_and_pointcloud(world, actor, dirname, timestamp, fp_town_points, actor_type, vehicle_id=None):
#     """
#     處理單個車輛的深度影像捕捉、點雲生成和特徵點比對。
#     """
#     # 深度相機捕捉
#     depth_camera = attach_camera_to_vehicle(world, actor, camera_type='sensor.camera.depth')
#     if not depth_camera:
#         print(f"{actor_type} {vehicle_id} 無法附加深度相機，跳過處理。")
#         return

#     depth_image_data = []

#     def save_depth_image(image):
#         depth_image_data.append(image)
#         depth_array = np.array(image.raw_data, dtype=np.float32).reshape((image.height, image.width, 4))[:, :, 0]
#         depth_array_normalized = (depth_array / np.max(depth_array) * 255.0).astype(np.uint8)
#         depth_image_path = f"{dirname}/depth_{timestamp}.png"
#         cv2.imwrite(depth_image_path, depth_array_normalized)
#         # print(f"灰階深度影像已保存於 {depth_image_path}。")

#     depth_camera.listen(save_depth_image)

#     # 等待影像捕捉完成
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if depth_image_data:
#             break

#     depth_camera.stop()

#     if not depth_image_data:
#         print(f"未捕捉到深度影像，跳過 {actor_type} {vehicle_id} 的處理。")
#         # depth_camera.destroy()
#     else:
#         depth_image = depth_image_data[0]
#         intrinsic = get_camera_intrinsic(depth_camera)
#         transform = depth_camera.get_transform()


#         # 將深度影像轉換為點雲
#         depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
#         points = []

#         for y in range(depth_array.shape[0]):
#             for x in range(depth_array.shape[1]):
#                 depth = depth_array[y, x]
#                 if depth <= 0:
#                     continue
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#                 world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
#                 points.append([int(world_coords.x), int(world_coords.y), int(world_coords.z)])

#         points = np.array(points)

#         # 與特徵點比對
#         matched_points = []
#         for point in points:
#             distances = np.linalg.norm(fp_town_points - point, axis=1)
#             if np.any(distances < 0.01):  # 距離閾值
#                 matched_points.append(point)

#         if len(matched_points) > 4:
#             print(f"{actor_type} {vehicle_id} 在時間 {timestamp} 的深度影像包含至少 4 個特徵點，符合條件。")

#             # 保存點雲
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_pointcloud_to_ply(points, ply_filename)

#             # 保存特徵點
#             matched_txt_filename = f"{dirname}/FP_{timestamp}.txt"
#             np.savetxt(matched_txt_filename, matched_points, fmt="%d", delimiter=" ")

#             # 計算重心
#             centroid = np.mean(matched_points, axis=0)
#             with open(matched_txt_filename, "a") as f:
#                 f.write(f"重心: {centroid[0]:.2f} {centroid[1]:.2f} {centroid[2]:.2f}\n")
#             print(f"重心座標: {centroid}")

#             # 切換到 RGB 相機並捕捉 RGB 影像
#             rgb_camera = attach_camera_to_vehicle(world, actor, camera_type='sensor.camera.rgb')
#             if rgb_camera:
#                 rgb_image_data = []

#                 def save_rgb_image(image):
#                     rgb_image_data.append(image)

#                 rgb_camera.listen(save_rgb_image)
#                 world.tick()
#                 time.sleep(0.5)  # 確保影像捕捉完成
#                 rgb_camera.stop()

#                 if rgb_image_data:
#                     rgb_image = rgb_image_data[0]
#                     rgb_image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")
#                     print(f"{actor_type} {vehicle_id} 在時間 {timestamp} 的 RGB 影像已保存。")

#                 rgb_camera.destroy()

#         else:
#             print(f"{actor_type} {vehicle_id} 在時間 {timestamp} 的深度影像不符合條件，跳過。")

#     # 銷毀相機
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()

# def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file):
#     """
#     捕捉 RGB 和深度影像，將 2D 特徵點轉換為 3D 世界座標，
#     並與 PCD_FP_town01.txt 中的點進行匹配。

#     Args:
#         world: CARLA 世界對象。
#         vehicle: 要附加相機的車輛。
#         dirname: 輸出檔案保存的目錄。
#         timestamp: 檔案命名的時間戳。
#         fp_town_file: 包含世界座標的 PCD_FP_town01.txt 的路徑。
#     """
#     # 附加相機
#     rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
#     depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

#     if not rgb_camera or not depth_camera:
#         print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加相機。")
#         if rgb_camera:
#             rgb_camera.destroy()
#         if depth_camera:
#             depth_camera.destroy()
#         return

#     # 加載 PCD_FP_town01.txt 中的點
#     try:
#         fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
#     except Exception as e:
#         print(f"載入 PCD_FP_town01.txt 錯誤：{e}")
#         return

#     # 用於捕捉數據的容器
#     rgb_image_data = []
#     depth_image_data = []

#     def save_rgb_image(image):
#         rgb_image_data.append(image)
#         image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")
#         # print(f"RGB 影像已保存於 {dirname}/rgb_{timestamp}.png。")

#     def save_depth_image(image):
#         depth_image_data.append(image)
#         depth_array = np.array(image.raw_data, dtype=np.float32).reshape((image.height, image.width, 4))[:, :, 0]
#         depth_array_normalized = (depth_array / np.max(depth_array) * 255.0).astype(np.uint8)
#         depth_image_path = f"{dirname}/depth_{timestamp}.png"
#         cv2.imwrite(depth_image_path, depth_array_normalized)
#         # print(f"灰階深度影像已保存於 {depth_image_path}。")

#     rgb_camera.listen(save_rgb_image)
#     depth_camera.listen(save_depth_image)

#     # 等待影像捕捉完成
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if rgb_image_data and depth_image_data:
#             break

#     # 停止監聽相機
#     rgb_camera.stop()
#     depth_camera.stop()

#     if not (rgb_image_data and depth_image_data):
#         print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
#     else:
#         # 預計算相機內參與轉換
#         intrinsic = get_camera_intrinsic(rgb_camera)
#         transform = rgb_camera.get_transform()

#         # 在 RGB 影像中檢測特徵點
#         rgb_array = np.array(rgb_image_data[0].raw_data).reshape((720, 1280, 4))[:, :, :3]
#         detected_fp = detect_feature_points(rgb_array)

#         # 將 2D 特徵點轉換為 3D 世界座標
#         feature_points_3d = []
#         depth_array = np.array(depth_image_data[0].raw_data, dtype=np.float32).reshape((720, 1280, 4))[:, :, 0]

#         for x, y in detected_fp:
#             try:
#                 depth = depth_array[int(y), int(x)]
#                 if depth <= 0:
#                     continue  # 跳過無效深度點

#                 # 轉換為相機座標
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth

#                 # 轉換為世界座標
#                 world_coords = transform.transform(carla.Location(x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]))
#                 feature_points_3d.append([world_coords.x, world_coords.y, world_coords.z])
#             except Exception as e:
#                 print(f"將 2D 特徵點 ({x}, {y}) 轉換為 3D 世界座標時出錯：{e}")

#         feature_points_3d = np.array(feature_points_3d)

#         # 與 PCD_FP_town01.txt 中的點進行匹配
#         match_count = sum(
#             1 for wp in feature_points_3d
#             if any(np.linalg.norm(wp - tp) < 5 for tp in fp_town_points)
#         )

#         # 保存每張影像的 3D 特徵點
#         np.savetxt(f"{dirname}/3D_FP_{timestamp}.txt", feature_points_3d, fmt="%.6f", delimiter=" ")
#         print(f"3D 特徵點已保存於 {dirname}/3D_FP_{timestamp}.txt。")

#         if match_count >= 4:
#             # 輸出匹配數量
#             print("")
#             print(f"影像滿足條件!{dirname.split('/')[-1]}: 時間 {timestamp} 的影像匹配了 {match_count} 個 PCD_FP_town01.txt 中的點，開始處理數據。")
#             print("")

#             # 保存灰階深度影像
#             save_depth_image(depth_image_data[0])

#             # 處理並保存特徵點
#             process_actor_for_fp_and_pointcloud(rgb_image_data[0], depth_image_data[0], intrinsic, transform, dirname, timestamp)

#             # 從深度影像生成並保存 .ply 文件
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_ply_file(depth_image_data[0], intrinsic, transform, ply_filename)

#             # 估計相機姿態
#             try:
#                 detected_fp_array = np.array(detected_fp, dtype=np.float32)
#                 fp_town_points_array = np.array(fp_town_points[:len(detected_fp)], dtype=np.float32)
#                 rvec, tvec = estimate_camera_pose_2d_3d(detected_fp_array, fp_town_points_array)

#                 with open(f"{dirname}/pose_{timestamp}.txt", "w") as pose_file:
#                     pose_file.write(f"旋轉向量：{rvec.tolist()}\n")
#                     pose_file.write(f"平移向量：{tvec.tolist()}\n")
#                 print(f"相機姿態已保存於 {dirname}/pose_{timestamp}.txt。")
#             except Exception as e:
#                 print(f"估計相機姿態時出錯：{e}")
#         else:
#             print(f"{dirname.split('/')[-1]}: 時間 {timestamp} 的影像不滿足條件，跳過。")

#     # 銷毀相機
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()




# def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file):
#     """
#     捕捉 RGB 和深度影像，將 2D 特徵點轉換為 3D 世界座標，
#     並與 FP_town01.txt 中的點進行匹配。
#     """
#     # 附加相機
#     rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
#     depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

#     if not rgb_camera or not depth_camera:
#         print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加相機。")
#         if rgb_camera:
#             rgb_camera.destroy()
#         if depth_camera:
#             depth_camera.destroy()
#         return

#     # 加載 FP_town01.txt 中的點
#     try:
#         fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
#     except Exception as e:
#         print(f"載入 {fp_town_file} 錯誤：{e}")
#         return

#     # 用於捕捉數據的容器
#     rgb_image_data = []
#     depth_image_data = []

#     def save_rgb_image(image):
#         rgb_image_data.append(image)

#     def save_depth_image(image):
#         depth_image_data.append(image)

#     rgb_camera.listen(save_rgb_image)
#     depth_camera.listen(save_depth_image)

#     # 等待影像捕捉完成
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if rgb_image_data and depth_image_data:
#             break

#     # 停止監聽相機
#     rgb_camera.stop()
#     depth_camera.stop()

#     if not (rgb_image_data and depth_image_data):
#         print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
#     else:
#         depth_image = depth_image_data[0]
#         intrinsic = get_camera_intrinsic(depth_camera)
#         transform = depth_camera.get_transform()

#         # print("相機內參數矩陣：")
#         # print(intrinsic)

#         # 將深度影像轉換為點雲
#         depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
#         if np.all(depth_array <= 0):
#             print("深度影像未包含有效數據，點雲生成失敗。")
#             return
#         points = []

#         matched_points = []  # 完整小數座標
#         matched_int_set = set()  # 用於檢查整數座標是否已匹配過

#         for y in range(depth_array.shape[0]):
#             for x in range(depth_array.shape[1]):
#                 depth = depth_array[y, x]
#                 if depth <= 0:
#                     continue
#                 # 將像素點轉換為世界座標
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#                 world_coords = transform.transform(carla.Location(
#                     x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]
#                 ))
#                 points.append([world_coords.x, world_coords.y, world_coords.z])  # 將完整的小數座標加入點雲
#                 # 匹配點
#                 int_point = (int(world_coords.x), int(world_coords.y), int(world_coords.z))

#                 # 檢查是否已經匹配過
#                 if int_point in matched_int_set:
#                     continue

#                 # 比對 FP_town01.txt
#                 distances = np.linalg.norm(fp_town_points - np.array(int_point), axis=1)
#                 if np.any(distances < 0.01):  # 距離閾值
#                     matched_points.append([world_coords.x, world_coords.y, world_coords.z])
#                     matched_int_set.add(int_point)  # 記錄整數值避免重複

#         # 確認 FP 數量是否大於 4
#         if len(matched_points) > 4:
#             print("")
#             print(f"這張圖適合當KF！{dirname.split('/')[-1]} 在時間 {timestamp} 的影像匹配了 {len(matched_points)} 個特徵點，開始保存數據...")
#             print("")

#             # 保存灰階深度影像
#             depth_array_normalized = (depth_array / depth_array.max() * 255).astype(np.uint8)
#             cv2.imwrite(f"{dirname}/depth_{timestamp}.png", depth_array_normalized)

#             # 保存 RGB 影像
#             rgb_image = rgb_image_data[0]
#             rgb_image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")

#             # 保存點雲
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_pointcloud_to_ply(points, ply_filename)
            

#             # 保存完整小數座標特徵點
#             matched_txt_filename = f"{dirname}/FP_{timestamp}.txt"
#             np.savetxt(matched_txt_filename, matched_points, fmt="%.6f", delimiter=" ")
#             print(f"完整特徵點已保存於 {matched_txt_filename}。")

#             # 保存去重的整數座標特徵點
#             matched_int_filename = f"{dirname}/FP_int_{timestamp}.txt"
#             matched_int_points = np.array(list(matched_int_set))  # 將 set 轉為 numpy 陣列
#             np.savetxt(matched_int_filename, matched_int_points, fmt="%d", delimiter=" ")
#             print(f"整數特徵點已保存於 {matched_int_filename}。")

#             # # 新增 EPnP 驗證
#             # intrinsic = np.array([[fx, 0, cx],
#             #                     [0, fy, cy],
#             #                     [0, 0, 1]])
#             validate_epnp(vehicle, matched_points, dirname, timestamp, intrinsic)

#         else:
#             print(f"時間 {timestamp} 的影像不滿足條件，跳過保存。")

#     # 銷毀相機
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()



# def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file, fp_town_new_file):
#     """
#     捕捉 RGB 和深度影像，將 2D 特徵點轉換為 3D 世界座標，
#     並保存匹配的特徵點到全局文件 (FP_town01.txt) 和單張影像文件。
#     """
#     # 附加相機
#     rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
#     depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

#     if not rgb_camera or not depth_camera:
#         print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加相機。")
#         if rgb_camera:
#             rgb_camera.destroy()
#         if depth_camera:
#             depth_camera.destroy()
#         return

#     # 加載 FP_town01.txt 中的點
#     try:
#         fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
#     except Exception as e:
#         print(f"載入 {fp_town_file} 錯誤：{e}")
#         return

#     # 用於捕捉數據的容器
#     rgb_image_data = []
#     depth_image_data = []

#     def save_rgb_image(image):
#         rgb_image_data.append(image)

#     def save_depth_image(image):
#         depth_image_data.append(image)

#     rgb_camera.listen(save_rgb_image)
#     depth_camera.listen(save_depth_image)

#     # 等待影像捕捉完成
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if rgb_image_data and depth_image_data:
#             break

#     # 停止監聽相機
#     rgb_camera.stop()
#     depth_camera.stop()

#     if not (rgb_image_data and depth_image_data):
#         print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
#     else:
#         depth_image = depth_image_data[0]
#         intrinsic = get_camera_intrinsic(depth_camera)
#         transform = depth_camera.get_transform()

#         # 將深度影像轉換為點雲
#         depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
#         if np.all(depth_array <= 0):
#             print("深度影像未包含有效數據，點雲生成失敗。")
#             return
#         points = []

#         matched_points = []  # 完整小數座標
#         matched_int_set = set()  # 用於檢查整數座標是否已匹配過

#         for y in range(depth_array.shape[0]):
#             for x in range(depth_array.shape[1]):
#                 depth = depth_array[y, x]
#                 if depth <= 0:
#                     continue
#                 # 將像素點轉換為世界座標
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#                 world_coords = transform.transform(carla.Location(
#                     x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]
#                 ))
#                 points.append([world_coords.x, world_coords.y, world_coords.z])  # 將完整的小數座標加入點雲
#                 # 匹配點
#                 int_point = (int(world_coords.x), int(world_coords.y), int(world_coords.z))

#                 # 檢查是否已經匹配過
#                 if int_point in matched_int_set:
#                     continue

#                 # 比對 FP_town01.txt
#                 distances = np.linalg.norm(fp_town_points - np.array(int_point), axis=1)
#                 if np.any(distances < 0.01):  # 距離閾值
#                     matched_points.append([world_coords.x, world_coords.y, world_coords.z])
#                     matched_int_set.add(int_point)  # 記錄整數值避免重複

#         # 保存到全局文件 fp_town_new_file.txt
#         with open(fp_town_new_file, "a") as f:
#             for int_point in matched_int_set:
#                 f.write(f"{int_point[0]} {int_point[1]} {int_point[2]}\n")

#         # 確認 FP 數量是否大於 4
#         if len(matched_points) > 4:
#             print(f"這張圖適合當KF！影像匹配了 {len(matched_points)} 個特徵點，開始保存數據...")

#             # 保存灰階深度影像
#             depth_array_normalized = (depth_array / depth_array.max() * 255).astype(np.uint8)
#             cv2.imwrite(f"{dirname}/depth_{timestamp}.png", depth_array_normalized)

#             # 保存 RGB 影像
#             rgb_image = rgb_image_data[0]
#             rgb_image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")

#             # 保存點雲
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_pointcloud_to_ply(points, ply_filename)

#             # 保存完整小數座標特徵點
#             matched_txt_filename = f"{dirname}/FP_{timestamp}.txt"
#             np.savetxt(matched_txt_filename, matched_points, fmt="%.6f", delimiter=" ")
#             print(f"完整特徵點已保存於 {matched_txt_filename}。")

#             # 保存去重的整數座標特徵點
#             matched_int_filename = f"{dirname}/FP_int_{timestamp}.txt"
#             matched_int_points = np.array(list(matched_int_set))  # 將 set 轉為 numpy 陣列
#             np.savetxt(matched_int_filename, matched_int_points, fmt="%d", delimiter=" ")
#             print(f"整數特徵點已保存於 {matched_int_filename}。")

#             # 新增 EPnP 驗證
#             validate_epnp(vehicle, matched_points, dirname, timestamp, intrinsic)
#         else:
#             print(f"時間 {timestamp} 的影像不滿足條件，跳過保存。")

#     # 銷毀相機
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()

# def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file, fp_town_new_file):
#     """
#     捕捉 RGB 和深度影像，將 2D 特徵點轉換為 3D 世界座標，
#     並保存匹配的特徵點到全局文件 (FP_town_new_file.txt) 和單張影像文件。
#     """
#     # 附加相機
#     rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
#     depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

#     if not rgb_camera or not depth_camera:
#         print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加相機。")
#         if rgb_camera:
#             rgb_camera.destroy()
#         if depth_camera:
#             depth_camera.destroy()
#         return

#     # 加載 FP_town01.txt 中的點
#     try:
#         fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
#     except Exception as e:
#         print(f"載入 {fp_town_file} 錯誤：{e}")
#         return

#     # 用於捕捉數據的容器
#     rgb_image_data = []
#     depth_image_data = []

#     def save_rgb_image(image):
#         rgb_image_data.append(image)

#     def save_depth_image(image):
#         depth_image_data.append(image)

#     rgb_camera.listen(save_rgb_image)
#     depth_camera.listen(save_depth_image)

#     # 等待影像捕捉完成
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if rgb_image_data and depth_image_data:
#             break

#     # 停止監聽相機
#     rgb_camera.stop()
#     depth_camera.stop()

#     if not (rgb_image_data and depth_image_data):
#         print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
#     else:
#         depth_image = depth_image_data[0]
#         intrinsic = get_camera_intrinsic(depth_camera)
#         transform = depth_camera.get_transform()

#         # 將深度影像轉換為點雲
#         depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]

#         # 增強深度數據對比
#         depth_array_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
#         depth_array_normalized *= 1000.0  # 調整至有效深度範圍（例如 0-1000 米）

#         if np.all(depth_array_normalized <= 0):
#             print("深度影像未包含有效數據，點雲生成失敗。")
#             return
#         points = []

#         matched_points = []  # 完整小數座標
#         matched_int_set = set()  # 用於檢查整數座標是否已匹配過

#         for y in range(depth_array_normalized.shape[0]):
#             for x in range(depth_array_normalized.shape[1]):
#                 depth = depth_array_normalized[y, x]
#                 if depth <= 0:
#                     continue
#                 # 將像素點轉換為世界座標
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#                 world_coords = transform.transform(carla.Location(
#                     x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]
#                 ))
#                 points.append([world_coords.x, world_coords.y, world_coords.z])  # 將完整的小數座標加入點雲
#                 # 匹配點
#                 int_point = (int(world_coords.x), int(world_coords.y), int(world_coords.z))

#                 # 檢查是否已經匹配過
#                 if int_point in matched_int_set:
#                     continue

#                 # 比對 FP_town01.txt
#                 distances = np.linalg.norm(fp_town_points - np.array(int_point), axis=1)
#                 if np.any(distances < 0.01):  # 距離閾值
#                     matched_points.append([world_coords.x, world_coords.y, world_coords.z])
#                     matched_int_set.add(int_point)  # 記錄整數值避免重複

#         # 保存到全局文件 fp_town_new_file.txt
#         with open(fp_town_new_file, "a") as f:
#             for int_point in matched_int_set:
#                 f.write(f"{int_point[0]} {int_point[1]} {int_point[2]}\n")

#         # 確認 FP 數量是否大於 4
#         if len(matched_points) > 4:
#             print(f"這張圖適合當KF！影像匹配了 {len(matched_points)} 個特徵點，開始保存數據...")

#             # 保存灰階深度影像
#             depth_array_gray = (depth_array_normalized / depth_array_normalized.max() * 255).astype(np.uint8)
#             cv2.imwrite(f"{dirname}/depth_{timestamp}.png", depth_array_gray)

#             # 保存 RGB 影像
#             rgb_image = rgb_image_data[0]
#             rgb_image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")

#             # 保存點雲
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_pointcloud_to_ply(points, ply_filename)

#             # 保存完整小數座標特徵點
#             matched_txt_filename = f"{dirname}/FP_{timestamp}.txt"
#             np.savetxt(matched_txt_filename, matched_points, fmt="%.6f", delimiter=" ")
#             print(f"完整特徵點已保存於 {matched_txt_filename}。")

#             # 保存去重的整數座標特徵點
#             matched_int_filename = f"{dirname}/FP_int_{timestamp}.txt"
#             matched_int_points = np.array(list(matched_int_set))  # 將 set 轉為 numpy 陣列
#             np.savetxt(matched_int_filename, matched_int_points, fmt="%d", delimiter=" ")
#             print(f"整數特徵點已保存於 {matched_int_filename}。")

#             # 新增 EPnP 驗證
#             validate_epnp(vehicle, matched_points, dirname, timestamp, intrinsic)
#         else:
#             print(f"時間 {timestamp} 的影像不滿足條件，跳過保存。")

#     # 銷毀相機
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()

# def capture_images_with_feature_points(world, vehicle, dirname, timestamp, fp_town_file, fp_town_new_file):
#     """
#     捕捉 RGB 和深度影像，將 2D 特徵點轉換為 3D 世界座標，
#     並保存匹配的特徵點到全局文件 (FP_town_new_file.txt) 和單張影像文件。
#     """
#     # 附加相機
#     rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
#     depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')
#     lidar_camera = attach_lidar_to_vehicle(world, vehicle)

#     if not rgb_camera or not depth_camera:
#         print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加相機。")
#         if rgb_camera:
#             rgb_camera.destroy()
#         if depth_camera:
#             depth_camera.destroy()
#         return

#     # 加載 FP_town01.txt 中的點
#     try:
#         fp_town_points = np.loadtxt(fp_town_file, delimiter=' ')
#     except Exception as e:
#         print(f"載入 {fp_town_file} 錯誤：{e}")
#         return

#     # 用於捕捉數據的容器
#     rgb_image_data = []
#     depth_image_data = []

#     def save_rgb_image(image):
#         rgb_image_data.append(image)

#     def save_depth_image(image):
#         depth_image_data.append(image)

#     rgb_camera.listen(save_rgb_image)
#     depth_camera.listen(save_depth_image)

#     # 等待影像捕捉完成
#     for _ in range(20):
#         world.tick()
#         time.sleep(0.1)
#         if rgb_image_data and depth_image_data:
#             break

#     # 停止監聽相機
#     rgb_camera.stop()
#     depth_camera.stop()

#     if not (rgb_image_data and depth_image_data):
#         print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
#     else:
#         depth_image = depth_image_data[0]
#         depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]

#         # 增強深度數據對比
#         depth_array_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255.0
#         depth_array_normalized = depth_array_normalized.astype(np.uint8)

#         # 使用 Matplotlib 顯示深度影像，3 秒後自動關閉
#         plt.imshow(depth_array_normalized, cmap='gray')
#         plt.title("Depth Image")
#         plt.axis('off')
#         plt.draw()  # 繪製影像
#         plt.pause(3)  # 停留 3 秒
#         plt.close()  # 關閉視窗

#         # 保存深度影像
#         plt.imsave(f"{dirname}/depth_{timestamp}.png", depth_array_normalized, cmap='gray')
#         print(f"深度影像已保存至 {dirname}/depth_{timestamp}.png。")

#         # 繼續執行後續邏輯
#         intrinsic = get_camera_intrinsic(depth_camera)
#         transform = depth_camera.get_transform()

#         if np.all(depth_array_normalized <= 0):
#             print("深度影像未包含有效數據，點雲生成失敗。")
#             return      
        
#         points = []

#         matched_points = []  # 完整小數座標
#         matched_int_set = set()  # 用於檢查整數座標是否已匹配過

#         for y in range(depth_array.shape[0]):
#             for x in range(depth_array.shape[1]):
#                 depth = depth_array[y, x]
#                 if depth <= 0:
#                     continue
#                 # 將像素點轉換為世界座標
#                 pixel_coords = np.array([x, y, 1.0])
#                 camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
#                 world_coords = transform.transform(carla.Location(
#                     x=camera_coords[0], y=camera_coords[1], z=camera_coords[2]
#                 ))
#                 points.append([world_coords.x, world_coords.y, world_coords.z])  # 將完整的小數座標加入點雲
#                 # 匹配點
#                 int_point = (int(world_coords.x), int(world_coords.y), int(world_coords.z))

#                 # 檢查是否已經匹配過
#                 if int_point in matched_int_set:
#                     continue

#                 # 比對 FP_town01.txt
#                 distances = np.linalg.norm(fp_town_points - np.array(int_point), axis=1)
#                 if np.any(distances < 0.01):  # 距離閾值
#                     matched_points.append([world_coords.x, world_coords.y, world_coords.z])
#                     matched_int_set.add(int_point)  # 記錄整數值避免重複

#         # 保存到全局文件 fp_town_new_file.txt
#         with open(fp_town_new_file, "a") as f:
#             for int_point in matched_int_set:
#                 f.write(f"{int_point[0]} {int_point[1]} {int_point[2]}\n")

#         print(f"成功保存匹配點至 {fp_town_new_file}")

#         # 確認 FP 數量是否大於 4
#         if len(matched_points) > 4:
#             print(f"這張圖適合當KF！影像匹配了 {len(matched_points)} 個特徵點，開始保存數據...")

#             # 保存灰階深度影像
#             depth_array_gray = (depth_array_normalized / depth_array_normalized.max() * 255).astype(np.uint8)
#             cv2.imwrite(f"{dirname}/depth_{timestamp}.png", depth_array_gray)

#             # 保存 RGB 影像
#             rgb_image = rgb_image_data[0]
#             rgb_image.save_to_disk(f"{dirname}/rgb_{timestamp}.png")

#             # 保存點雲
#             ply_filename = f"{dirname}/pointcloud_{timestamp}.ply"
#             save_pointcloud_to_ply(points, ply_filename)

#             # 保存完整小數座標特徵點
#             matched_txt_filename = f"{dirname}/FP_{timestamp}.txt"
#             np.savetxt(matched_txt_filename, matched_points, fmt="%.6f", delimiter=" ")
#             print(f"完整特徵點已保存於 {matched_txt_filename}。")

#             # 保存去重的整數座標特徵點
#             matched_int_filename = f"{dirname}/FP_int_{timestamp}.txt"
#             matched_int_points = np.array(list(matched_int_set))  # 將 set 轉為 numpy 陣列
#             np.savetxt(matched_int_filename, matched_int_points, fmt="%d", delimiter=" ")
#             print(f"整數特徵點已保存於 {matched_int_filename}。")

#             # 新增 EPnP 驗證
#             validate_epnp(vehicle, matched_points, dirname, timestamp, intrinsic)
#         else:
#             print(f"時間 {timestamp} 的影像不滿足條件，跳過保存。")

#     # 銷毀相機
#     if rgb_camera.is_alive:
#         rgb_camera.destroy()
#     if depth_camera.is_alive:
#         depth_camera.destroy()

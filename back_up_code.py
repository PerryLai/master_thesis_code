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


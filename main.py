# main.py
import carla
import os
import shutil
from satellite_module import capture_satellite_images
from simulation_module import generate_simulation_data  # Unified function for vehicles and walkers

def main():
    # Carla 連線
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    
    # 衛星影像保存目錄
    map_image_dir = "data/map_image"
    if os.path.exists(map_image_dir):
        shutil.rmtree(map_image_dir)
    os.makedirs(map_image_dir, exist_ok=True)
    
    # 拍攝衛星影像
    print("Capturing satellite images...")
    map_center = carla.Location(x=0, y=0, z=0)  # 假設城市中心位置
    # capture_satellite_images(world, map_center, map_image_dir)
    print("Capturing satellite images finished!")
    
    # 開始模擬實驗
    num_experiments = 10
    for i in range(num_experiments):
        print(f"Running experiment {i}...")
        experiment_dir = f"data/{i}"
        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 生成車輛和行人數據
        print("Generating simulation data (vehicles and walkers)...")
        generate_simulation_data(world, i, experiment_dir, duration=10)
    
    print("All experiments completed!")

if __name__ == "__main__":
    main()

# main.py
# import carla
# import os
# import shutil
# from satellite_module import capture_satellite_images
# # from vehicle_module import generate_vehicle_data
# from vehicle_module_efficiency_ver import generate_vehicle_data
# # from walker_module import generate_walker_data
# from walker_module_efficiency_ver import generate_walker_data

# def main():
#     # Carla 連線
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(30.0)
#     world = client.get_world()
    
#     # 衛星影像保存目錄
#     map_image_dir = "data/map_image"
#     if os.path.exists(map_image_dir):
#         shutil.rmtree(map_image_dir)
#     os.makedirs(map_image_dir, exist_ok=True)
    
#     # 拍攝衛星影像
#     print("Capturing satellite images...")
#     map_center = carla.Location(x=0, y=0, z=0)  # 假設城市中心位置
#     # capture_satellite_images(world, map_center, map_image_dir)
#     print("Capturing satellite images finish!")
    
#     # 開始模擬實驗
#     num_experiments = 10
#     for i in range(num_experiments):
#         print(f"Running experiment {i}...")
#         experiment_dir = f"data/{i}"
#         if os.path.exists(experiment_dir):
#             shutil.rmtree(experiment_dir)
#         os.makedirs(experiment_dir, exist_ok=True)
        
#         # 生成行人數據
#         print("Generating walker data...")
#         generate_walker_data(world, i, experiment_dir)

#         # 生成車輛數據
#         print("Generating vehicle data...")
#         generate_vehicle_data(world, i, experiment_dir)
    
#     print("All experiments completed!")

# if __name__ == "__main__":
#     main()

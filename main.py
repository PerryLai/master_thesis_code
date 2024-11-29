# main.py
import carla
import os
import shutil
import numpy as np
# from satellite_module import capture_satellite_images
# from FPcollector import generate_simulation_data  # 這是為了生成FP_town01.txt用的
from datacollector import generate_simulation_data
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def main():
    # Carla 連線
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    # client.set_log_level(carla.LOG_LEVEL_ERROR)
    world = client.get_world()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # # 衛星影像保存目錄
    # map_image_dir = "data/map_image"
    # if os.path.exists(map_image_dir):
    #     shutil.rmtree(map_image_dir)
    # os.makedirs(map_image_dir, exist_ok=True)
    
    # 拍攝衛星影像
    # print("Capturing satellite images...")
    # map_center = carla.Location(x=0, y=0, z=0)  # 假設城市中心位置
    # capture_satellite_images(world, map_center, map_image_dir)
    # print("Capturing satellite images finished!")
    
    # 開始模擬實驗
    num_experiments = 2 # 重新部署人車分佈次數
    duration = 10       # 每輪的持續秒數
    vehicle_num = 50   # 設定城市裡有多少台車
    select_num = 5     # 每次監控多少台車並取得資料
    walker_num = 100    # 設定城市裡有多少行人

    fp_town_file = "output/FP_town01.txt"
    fp_town_new_file = "output/FP_town01_new.txt"

    for i in range(num_experiments):
        print(f"正在執行實驗 {i}...")
        experiment_dir = f"data/{i}"
        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 生成車輛和行人數據
        print("Generating simulation data (vehicles and walkers)...")
        generate_simulation_data(world, i, experiment_dir, duration, vehicle_num, walker_num, select_num, fp_town_file, fp_town_new_file)

        # 執行
    
    print("所有實驗已完成！")

if __name__ == "__main__":
    main()
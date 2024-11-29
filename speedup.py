import carla
import time

def move_spectator(world, direction, speed, duration):
    """
    移動 Spectator 第一人稱視角。

    Args:
        world: CARLA 世界對象。
        direction: 移動的方向向量 (carla.Vector3D)。
        speed: 移動速度 (單位: 米/秒)。
        duration: 移動的持續時間 (單位: 秒)。
    """
    spectator = world.get_spectator()
    start_transform = spectator.get_transform()

    # 計算每步的移動增量
    delta_distance = speed * 0.1  # 每 0.1 秒移動的距離
    move_vector = carla.Vector3D(
        x=direction.x * delta_distance,
        y=direction.y * delta_distance,
        z=direction.z * delta_distance
    )

    # 模擬移動
    for _ in range(int(duration / 0.1)):
        current_transform = spectator.get_transform()
        new_location = current_transform.location + move_vector
        current_transform.location = new_location
        spectator.set_transform(current_transform)
        time.sleep(0.1)

    print("Spectator moved successfully.")

# 示例使用
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 定義移動方向和速度
move_direction = carla.Vector3D(0, 1, 0)  # 向 X 軸正方向移動
move_speed = 500.0  # 每秒移動 5 米
move_duration = 3.0  # 持續移動 2 秒

move_spectator(world, move_direction, move_speed, move_duration)

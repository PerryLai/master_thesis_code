import carla
import random
import os
from carlaAPI.image_converter import depth_to_local_point_cloud, to_rgb_array
import carla
import os
from util import install_camera_set_for_car_model
import time
import json
import numpy as np
import open3d as o3d

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
world = client.get_world()
vehicle_blueprints = world.get_blueprint_library().filter('vehicle.dodge.charger_2020')
path = os.path.join('car_model')
spawn_points = world.get_map().get_spawn_points()

for i, blueprint in enumerate(vehicle_blueprints):
    print(i)
    if blueprint.get_attribute('number_of_wheels').as_int() != 4:
        continue

    ego_vehicle_blueprint = blueprint
    print(ego_vehicle_blueprint)
    ego_vehicle_spawn_point = spawn_points[0]
    print(ego_vehicle_spawn_point)
    ego_vehicle = world.spawn_actor(ego_vehicle_blueprint, ego_vehicle_spawn_point)
    box = ego_vehicle.bounding_box

    # ego_json = {'id': ego_vehicle_blueprint.id,
    #             'x': ego_vehicle_spawn_point.location.x,
    #             'y': ego_vehicle_spawn_point.location.y,
    #             'z': ego_vehicle_spawn_point.location.z,
    #             'pitch': ego_vehicle_spawn_point.rotation.pitch,
    #             'yaw': ego_vehicle_spawn_point.rotation.yaw,
    #             'roll': ego_vehicle_spawn_point.rotation.roll,
    #             'box_x': box.extent.x*2,
    #             'box_y': box.extent.y*2,
    #             'box_z': box.extent.z*2}


    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    box_x = box.extent.x*2
    box_y = box.extent.y*2
    box_z = box.extent.z*2
    cameras1 = install_camera_set_for_car_model(world, ego_vehicle, os.path.join(path, "tmp"), box_z+0.6)

    world.tick()
    print("Sleep 2s")
    time.sleep(2)
    plypath = os.path.join(path, "tmp")
    mesh1 = o3d.io.read_point_cloud(os.path.join(plypath,"1.ply"))
    mesh2 = o3d.io.read_point_cloud(os.path.join(plypath,"2.ply"))
    mesh3 = o3d.io.read_point_cloud(os.path.join(plypath,"3.ply"))
    mesh4 = o3d.io.read_point_cloud(os.path.join(plypath,"4.ply"))

    point1 = np.asarray(mesh1.points)
    point2 = np.asarray(mesh2.points)
    point3 = np.asarray(mesh3.points)
    point4 = np.asarray(mesh4.points)

    
    start_x = box_y/2
    end_x=-box_y/2
    start_y = box_x/2
    end_y=-box_x/2
    start_z = 0
    end_z=-box_z
    
    point1 = point1[point1[:,0] < start_x]
    point1 = point1[point1[:,0] > end_x]
    point1 = point1[point1[:,1] < 8+start_y]
    point1 = point1[point1[:,1] > 8+end_y]
    point1 = point1[point1[:,2] < start_z]
    point1 = point1[point1[:,2] > end_z]
    point1[:,1] = point1[:,1]-8
    
    
    point2[:, 1] = -point2[:, 1]
    point2[:, 0] = -point2[:, 0]
    point2 = point2[point2[:,0] < start_x]
    point2 = point2[point2[:,0] > end_x]
    point2 = point2[point2[:,1] > -(8+start_y)]
    point2 = point2[point2[:,1] < -(8+end_y)]
    point2 = point2[point2[:,2] < start_z]
    point2 = point2[point2[:,2] > end_z]
    point2[:,1] = point2[:,1]+8
    
    point3[:, [0,1]] = point3[:, [1,0]]
    point3[:, 1] = -point3[:, 1]
    point3 = point3[point3[:,0] > 5-start_x]
    point3 = point3[point3[:,0] < 5-end_x]
    point3 = point3[point3[:,1] < start_y]
    point3 = point3[point3[:,1] > end_y]
    point3 = point3[point3[:,2] < start_z]
    point3 = point3[point3[:,2] > end_z]
    point3[:,0] = point3[:,0]-5

    point4[:, [0,1]] = point4[:, [1,0]]
    point4[:, 0] = -point4[:, 0]
    point4 = point4[point4[:,0] < -(5-start_x)]
    point4 = point4[point4[:,0] > -(5-end_x)]
    point4 = point4[point4[:,1] < start_y]
    point4 = point4[point4[:,1] > end_y]
    point4 = point4[point4[:,2] < start_z]
    point4 = point4[point4[:,2] > end_z]
    point4[:,0] = point4[:,0]+5

    points = np.concatenate((point1, point2, point3, point4), axis=0)
    thresh = -box_z-0.56
    points = points[np.where(points[:,2] > thresh)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(path, "{}.ply").format(blueprint.id), pcd)
    world = client.reload_world()


import os
import tarfile
import gzip
import json
import numpy as np

from glob import glob
from scipy.spatial.transform import Rotation as R

from pprint import pprint

TSTEP = 0.1


def utm_to_bev(
        pt_utm_array: np.ndarray,
        ego_x_utm: float,
        ego_y_utm: float,
        ego_heading_utm: float
    ) -> np.ndarray:
    """Transform list of UTM points to BEV coordinate.

    Args:
        pt_utm_array (np.ndarray): List of UTM points. [[x, y, heading(Optional)]]
        ego_x_utm (float): Ego car x w.r.t. UTM
        ego_y_utm (float): Ego car y w.r.t. UTM
        ego_heading_utm (float): Ego car heading w.r.t. UTM

    Returns:
        np.ndarray: List of BEV points
    """
    
    is_single_pt = len(pt_utm_array.shape) == 1
    
    if is_single_pt:
        pt_utm_array = np.expand_dims(pt_utm_array, axis=0)

    is_heading_provided = len(pt_utm_array[0]) > 2

    pt_bev_array = []
    for i in range(len(pt_utm_array)):
        pt_utm_x = pt_utm_array[i][0]
        pt_utm_y = pt_utm_array[i][1]

        pt_x_shifted = pt_utm_x - ego_x_utm
        pt_y_shifted = pt_utm_y - ego_y_utm

        pt_x_rotated = pt_x_shifted * np.cos(ego_heading_utm) \
                        + pt_y_shifted * np.sin(ego_heading_utm)
        pt_y_rotated = - pt_x_shifted * np.sin(ego_heading_utm) \
                        + pt_y_shifted * np.cos(ego_heading_utm)
        
        if is_heading_provided:
            pt_utm_heading = pt_utm_array[i][2]
            pt_heading_rotated = pt_utm_heading - ego_heading_utm
            pt_bev_array.append([pt_x_rotated, pt_y_rotated, pt_heading_rotated])
        else:
            pt_bev_array.append([pt_x_rotated, pt_y_rotated])

    pt_bev_array = np.asarray(pt_bev_array)
    
    if is_single_pt:
        return pt_bev_array[0]
    else:
        return pt_bev_array

def form_frame_template(timestamp: float, idx: int, num_frame: int):
    frame = {
        "prev": None if idx == 0 else str(timestamp + TSTEP * (idx - 1)),
        "next": None if idx == num_frame - 1 else str(timestamp + TSTEP * (idx + 1)),
        "timestamp": str(timestamp + TSTEP * idx),
        "navi": None,
        "turn_switch": -1,
        "can_bus": None,
        "cams": None,
        "lidar2ego_translation": None,
        "lidar2ego_rotation": None,
        "ego2global_translation": None,
        "ego2global_rotation": None,
        "agents": None,
        "map": None
    }
    return frame

def find_ego_box(boxes):
    for box in boxes:
        if box["class"] == "ego_vehicle":
            return box

def extract_can_bus(anno):
    """ From ego bounding box
    """

    ego_box = find_ego_box(anno["bounding_boxes"])

    position = ego_box["location"]
    rot_matrix = R.from_euler("xyz", ego_box["rotation"], degrees=True)
    quat = rot_matrix.as_quat().tolist()
    acceleration = anno["acceleration"]
    angular_velocity = anno["angular_velocity"]
    velocity = (anno["speed"] * rot_matrix.apply(np.array([1, 0, 0]))).tolist()
    heading_rad = np.deg2rad(ego_box["rotation"][-1])
    heading_deg = ego_box["rotation"][-1]

    can_bus = position \
                + quat \
                + acceleration \
                + angular_velocity \
                + velocity \
                + [heading_rad, heading_deg]

    return can_bus

def extract_agents(anno, MAX_DISTANCE=100, FILTER_Z_SHRESHOLD=10):
    ego_box = find_ego_box(anno["bounding_boxes"])
    ego_x = ego_box["location"][0]
    ego_y = ego_box["location"][1]
    ego_heading = np.deg2rad(ego_box["rotation"][-1])

    agents = {}
    for npc in anno['bounding_boxes']:
        if npc['class'] in ['ego_vehicle', "traffic_sign"]:
            continue
        if npc['distance'] > MAX_DISTANCE:
            continue
        if abs(npc['location'][2] - ego_box['location'][2]) > FILTER_Z_SHRESHOLD:
            continue

        # pose in global
        location_global = npc["center"]
        yaw_global = np.deg2rad(npc["rotation"][-1])
        world2ego = R.from_euler("xyz", ego_box["rotation"], degrees=True)
        if npc['class'] in ["traffic_light", "traffic_sign"]:
            speed = 0.0
        else:
            speed = npc["speed"]
        velocity_global = (speed * world2ego.apply(np.array([1, 0, 0]))).tolist()
        acceleration_global = [0.0, 0.0, 0.0]

        # pose in local
        pose_local = utm_to_bev(
            np.array(location_global[0:2] + [yaw_global]),
            ego_x, ego_y, ego_heading)
        location_local = pose_local[0:2].tolist()
        yaw_local = pose_local[2]
        velocity_local = world2ego.apply(velocity_global).tolist()
        acceleration_local = [0.0, 0.0, 0.0]

        # type
        CLASS_TO_TYPE = {
            "vehicle": 1,
            "traffic_light": 14,
            "walker": 8
        }
        type = CLASS_TO_TYPE[npc["class"]]

        # is_movable
        if npc["class"] == "vehicle":
            is_movable = npc["state"] == "dynamic"
        if npc["class"] == "traffic_light":
            is_movable = False
        if npc["class"] == "walker":
            is_movable = True
        
        # dimension
        length, width, height = np.array(npc["extent"]) * 2

        # fill
        agents[npc["id"]] = {
            "id": npc["id"],
            "pose": {
                "ego": {
                    "position": location_local,
                    "heading": yaw_local,
                    "velocity": velocity_local,
                    "acceleration": acceleration_local # all 0, no valid value
                },
                "odom": {
                    "position": location_global,
                    "heading": yaw_global,
                    "velocity": velocity_global,
                    "acceleration": acceleration_global
                }
            },
            "type": type,
            "is_movable": is_movable,
            "dimension": {
                "corner_points": [], # empty, calculate if needed
                "width": width,
                "height": height,
                "length": length
            }
        }

    return agents

def extract_from_one_tar_file(tar_file: str):
    
    # Read clip metadata
    folder = os.path.dirname(tar_file)
    scene = os.path.basename(tar_file).split(".")[0]
    scenario, map, route, weather = scene.split("_")

    # Extract .tar.gz
    if not os.path.isdir(os.path.join(folder, scene)):
        print(f"Extracting {tar_file}...")
        with tarfile.open(tar_file) as f:
            f.extractall(folder)

    # Anno
    anno_file_list = glob(os.path.join(folder, scene, "anno", "*.json.gz"))
    anno_file_list.sort()
    # for idx_anno, anno_file in enumerate(anno_file_list):
    for idx_anno, anno_file in enumerate(anno_file_list[12:13]):
        with gzip.open(anno_file, 'rb') as f:
            anno = json.load(f)

        frame = form_frame_template(19980518.0, idx_anno, len(anno_file_list))
        frame["can_bus"] = extract_can_bus(anno)
        frame["agents"] = extract_agents(anno)

        pprint(frame)

    return


if __name__ == "__main__":

    DATA_ROOT = "../data/Bench2Drive-mini"

    tar_file_list = glob(os.path.join(DATA_ROOT, "*.tar.gz"))
    for tar_file in tar_file_list:
        extract_from_one_tar_file(tar_file)
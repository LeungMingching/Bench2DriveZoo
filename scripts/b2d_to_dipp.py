import os
import time
import pickle
import tarfile
import gzip
import json
import numpy as np

from copy import deepcopy
from glob import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from pprint import pprint

TSTEP = 100 # ms


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

def get_default_frame(timestamp: float, idx: int, num_frame: int):
    frame = {
        "prev": None if idx == 0 else str(timestamp + TSTEP * (idx - 1)).split(".")[0],
        "next": None if idx == num_frame - 1 else str(timestamp + TSTEP * (idx + 1)).split(".")[0],
        "timestamp": str(timestamp + TSTEP * idx).split(".")[0],
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
        location_local = pose_local[0:2].tolist() + [0.0]
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
        corner_points_in_ag_cord = [
            [0.5 * length, 0.5 * width, 0],
            [0.5 * length, - 0.5 * width, 0],
            [- 0.5 * length, - 0.5 * width, 0],
            [- 0.5 * length, 0.5 * width, 0]]
        trans_ego2ag = np.array([pose_local[0], pose_local[1], 0])
        rot_ego2ag = R.from_euler("xyz", [0 ,0, yaw_local])
        corner_points = [
            (rot_ego2ag.apply(np.array(pt)) + trans_ego2ag).tolist() for pt in corner_points_in_ag_cord]

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
                "corner_points": corner_points, # empty, calculate if needed
                "width": width,
                "height": height,
                "length": length
            }
        }

    return agents

def reconstruct_lane_line(line):

    lane_line = {
        "id": None,
        "road_id": None,
        "type": [],
        "separate_index": [],
        "point": {
            "odom": [],
            "ego": []
        }
    }

    return lane_line

def reconstruct_stop_line(line):



    stop_line = {
        "id": None,
        "road_id": None,
        "type": None,
        "point": {
            "odom": [],
            "ego": []
        }
    }

    return stop_line

def reconstruct_curbs(line):



    curbs = {
        "id": None,
        "road_id": None,
        "point": {
            "odom": [],
            "ego": []
        }
    }

    return curbs

def find_lanes_of_interest(map, ego_road_id, ego_lane_id):
    lanes_of_interest = [(ego_road_id, ego_lane_id)]

    for line in map[ego_road_id][ego_lane_id]:
        if line["Type"] == "Center":
            left_lane = line["Left"]
            right_lane = line["Right"]
            if (left_lane) \
                and (left_lane not in lanes_of_interest) \
                and (left_lane[1] in map[ego_road_id].keys()):
                lanes_of_interest.append(left_lane)
            if (right_lane) \
                and (right_lane not in lanes_of_interest) \
                and (right_lane[1] in map[ego_road_id].keys()):
                lanes_of_interest.append(right_lane)

    return lanes_of_interest

def extract_maps(anno, map_dict):
    LANE_TYPE_MAPPING = {
        ("White", "Broken"): 1,
        ("White", "Solid"): 2,
        ("White", "SolidSolid"): 4,
        ("Yellow", "Broken"): 7,
        ("Yellow", "Solid"): 8,
        ("Yellow", "SolidSolid"): 10
    }
    
    result_map_dict = {
        "map_type": "carla",
        "reference_lines": [],
        "lane_lines": [],
        "stop_lines": [],
        "curbs": []
    }
    
    ego_box = find_ego_box(anno["bounding_boxes"])

    # determin which lane should be extracted
    lanes_of_interest = find_lanes_of_interest(
                            map_dict, ego_box["road_id"], ego_box["lane_id"])
    
    for road_id, lane_id in lanes_of_interest:
        
        # Traffic light or Stop sign
        if "Trigger_Volumes" in map_dict[road_id].keys():
            for trigger_volume in map_dict[road_id]["Trigger_Volumes"]:
                # TODO: do something to trigger volumes
                continue

        for line in map_dict[road_id][lane_id]:
            points_global = np.array([[pt[0], pt[1]] for pt in np.array(line["Points"], dtype=object)[:, 0]])
            points_local = utm_to_bev(points_global, 
                            ego_box["location"][0], ego_box["location"][1], np.deg2rad(ego_box["rotation"][-1]))
            if line["Type"] == "Center":
                reference_line = {
                    "id": lane_id,
                    "lane_attribute": None,
                    "passable_type": None,
                    "point": {
                        "odom": points_global.tolist(),
                        "ego": points_local.tolist()
                    },
                    "road_id": road_id,
                    "left_lane_line_id": None,
                    "right_lane_line_id": None,
                    "left_neighbour_id": line["Left"][1],
                    "right_neighbour_id": line["Right"][1]
                }
                result_map_dict["reference_lines"].append(deepcopy(reference_line))
            else:
                converted_type = LANE_TYPE_MAPPING.get((line["Type"], line["Color"]), 1)
                lane_line = {
                    "id": lane_id,
                    "road_id": road_id,
                    "type": [converted_type],
                    "separate_index": [],
                    "point": {
                        "odom": points_global.tolist(),
                        "ego": points_local.tolist()
                    }
                }
                result_map_dict["lane_lines"].append(deepcopy(lane_line))

    return result_map_dict

def extract_from_one_tar_file(tar_file: str):
    frame_list = []
    
    # Read clip metadata
    folder = os.path.dirname(tar_file)
    scene = os.path.basename(tar_file).split(".")[0]
    scenario, map_name, route, weather = scene.split("_")

    # Extract .tar.gz
    if not os.path.isdir(os.path.join(folder, scene)):
        print(f"Extracting {tar_file}...")
        with tarfile.open(tar_file) as f:
            f.extractall(folder)

    # Anno
    anno_file_list = glob(os.path.join(folder, scene, "anno", "*.json.gz"))
    anno_file_list.sort()
    # for idx_anno, anno_file in enumerate(tqdm(anno_file_list)):
    for idx_anno, anno_file in enumerate(tqdm(anno_file_list[12:22])):
        with gzip.open(anno_file, 'rb') as f:
            anno = json.load(f)

        frame = get_default_frame(time.time() * 1e3, idx_anno, len(anno_file_list))
        frame["can_bus"] = extract_can_bus(anno)
        frame["agents"] = extract_agents(anno)
        
        map_file = os.path.join(folder, "maps", f"{map_name}_HD_map.npz")
        map = dict(np.load(map_file, allow_pickle=True)["arr"])
        frame["map"] = extract_maps(anno, map)

        # pprint(frame)
        frame_list.append(deepcopy(frame))
    return frame_list


if __name__ == "__main__":

    DATA_ROOT = "../data/Bench2Drive-mini"
    SAVE_ROOT = "../data/Bench2Drive-DIPP"

    tar_file_list = glob(os.path.join(DATA_ROOT, "*.tar.gz"))
    for tar_file in tar_file_list:
        frame_list = extract_from_one_tar_file(tar_file)
        
        scene = {
            "metadata": {
                "version": "v0.0",
                "scene_id": None,
                "desc": os.path.basename(tar_file).split(".")[0]
            },
            "infos": deepcopy(frame_list)
        }

        timestamp = scene["infos"][0]["timestamp"]
        if not(os.path.exists(os.path.join(SAVE_ROOT, timestamp))):
            os.makedirs(os.path.join(SAVE_ROOT, timestamp, "imgs"), exist_ok=True)
        # pickle
        with open(os.path.join(SAVE_ROOT, timestamp, timestamp + ".pkl"), "wb") as f:
            pickle.dump(scene, f)
        print("Saved to ", os.path.join(SAVE_ROOT, timestamp, timestamp + ".pkl"))
        # json
        with open(os.path.join(SAVE_ROOT, timestamp, timestamp + ".json"), "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, indent=4)
        print("Saved to ", os.path.join(SAVE_ROOT, timestamp, timestamp + ".json"))



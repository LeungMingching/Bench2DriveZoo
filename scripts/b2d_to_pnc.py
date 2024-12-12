import os
import time
import pickle
import tarfile
import gzip
import json
import shutil
import numpy as np

from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R


TSTEP = 100 # ms

DATA_ROOT = "../../data/Bench2Drive"
MAP_ROOT = os.path.join(DATA_ROOT, "maps")
SAVE_ROOT = "../../data/Bench2Drive-PnC"
NUM_WORKER = 1
SKIP_LIST = [
    
]
TEST_LIST = [
    
]


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

    rot_mat = np.array([
        [np.cos(ego_heading_utm), - np.sin(ego_heading_utm)],
        [np.sin(ego_heading_utm), np.cos(ego_heading_utm)]
    ])
    trans_mat = np.array([[ego_x_utm, ego_y_utm]]) \
                    .repeat(len(pt_utm_array), axis=0)

    pt_bev_position = np.dot(
        (pt_utm_array[:, :2] - trans_mat), rot_mat)
        
    if is_heading_provided:
        pt_bev_heading = pt_utm_array[:, 2] - ego_heading_utm
        pt_bev_array = np.concatenate(
            (pt_bev_position, np.expand_dims(pt_bev_heading, axis=-1)), axis=-1)
    else:
        pt_bev_array = pt_bev_position
    
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
    rotation = R.from_euler("xyz", ego_box["rotation"], degrees=True)
    quat = rotation.as_quat()[[3, 0, 1, 2]].tolist()
    acceleration = anno["acceleration"]
    angular_velocity = anno["angular_velocity"]
    velocity = (anno["speed"] * rotation.apply(np.array([1, 0, 0]))).tolist()
    heading_rad = np.deg2rad(ego_box["rotation"][-1])
    heading_deg = ego_box["rotation"][-1]

    can_bus = position \
                + quat \
                + acceleration \
                + angular_velocity \
                + velocity \
                + [heading_rad, heading_deg]

    ego2global_translation = position
    ego2global_rotation = rotation.as_matrix().tolist()

    return can_bus, (ego2global_translation, ego2global_rotation)

def extract_agents(anno, MAX_DISTANCE=100, FILTER_Z_SHRESHOLD=10):
    ego_box = find_ego_box(anno["bounding_boxes"])
    ego_x = ego_box["location"][0]
    ego_y = ego_box["location"][1]
    ego_heading = np.deg2rad(ego_box["rotation"][-1])

    agents = {}
    for npc in anno['bounding_boxes']:
        if npc['class'] in ['ego_vehicle']:
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
            "walker": 8,
            "traffic_light": 14,
            "traffic_sign": 10 # 水马等障碍物
        }
        type = CLASS_TO_TYPE[npc["class"]]

        # is_movable (default: False)
        is_movable = False
        if npc["class"] == "vehicle":
            is_movable = npc["state"] == "dynamic"
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
                "corner_points": corner_points,
                "width": width,
                "height": height,
                "length": length
            }
        }

    return agents

def find_n_level_lanes(map, road_id, lane_id, max_level, allow_junction=False):
    queue = deque([(road_id, lane_id, 0)]) # level 0
    visited = set()
    lane_list = []

    while queue:
        ptr_road_id, ptr_lane_id, level = queue.popleft()

        # reach max_level
        if level > max_level:
            break

        # visited
        if (ptr_road_id, ptr_lane_id) in visited:
            continue
        visited.add((ptr_road_id, ptr_lane_id))

        # get line
        reference_line = None
        for line in map[ptr_road_id][ptr_lane_id]:
            if line["Type"] == "Center":
                reference_line = line
        if not reference_line:
            continue

        if not allow_junction and reference_line["TopologyType"] == "Junction":
            for next_road_id, next_lane_id in reference_line["Topology"]:
                if (next_road_id, next_lane_id) not in visited:
                    queue.append((next_road_id, next_lane_id, level))
        else:
            lane_list.append((ptr_road_id, ptr_lane_id))

            # Left, Right Lanes
            left_lane = reference_line["Left"]
            right_lane = reference_line["Right"]
            if (left_lane) \
                and (left_lane not in lane_list) \
                and (left_lane[1] in map[road_id].keys()):
                lane_list.append(left_lane)
            if (right_lane) \
                and (right_lane not in lane_list) \
                and (right_lane[1] in map[road_id].keys()):
                lane_list.append(right_lane)

            for next_road_id, next_lane_id in reference_line["Topology"]:
                if (next_road_id, next_lane_id) not in visited:
                    queue.append((next_road_id, next_lane_id, level + 1))

    return lane_list

def extract_maps(anno, map_dict, max_num_points=200):
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
    lanes_of_interest = find_n_level_lanes(
                            map_dict, ego_box["road_id"], ego_box["lane_id"], 1, allow_junction=False)
    
    # Traffic lights
    passable_type_mapping = {}
    for box in anno["bounding_boxes"]:
        if box["class"] == "traffic_light": 
            if box["state"] == 0:
                passable_type_mapping[(box["road_id"], box["lane_id"])] = int(np.uint8(0b00000000))
            else:
                passable_type_mapping[(box["road_id"], box["lane_id"])] = int(np.uint8(0b11111111))

    # Reference lines or Lane lines
    def create_ROI_mask(bev_points,
        max_lon=300, min_lon=-100,
        max_lat=80, min_lat=-80,
        max_height=15, min_height=-15
    ):
        assert bev_points.shape[1] == 3, "Not 3D-Point"
        upper = np.array([[max_lon, max_lat, max_height]]).repeat(len(bev_points), axis=0)
        lower = np.array([[min_lon, min_lat, min_height]]).repeat(len(bev_points), axis=0)
        return (bev_points <= upper) & (bev_points > lower)

    for road_id, lane_id in lanes_of_interest:

        # protect non-exist id
        if road_id not in map_dict.keys():
            continue
        if lane_id not in map_dict[road_id].keys():
            continue

        for line in map_dict[road_id][lane_id]:
            points_global = np.array([[pt[0], pt[1], 0.0] for pt in np.array(line["Points"], dtype=object)[:, 0]])
            points_local = utm_to_bev(points_global[:, :2], 
                            ego_box["location"][0], ego_box["location"][1], np.deg2rad(ego_box["rotation"][-1]))
            points_local = np.concatenate((points_local, np.zeros((len(points_local), 1))), axis=-1)

            # ROI masking
            points_global = np.where(create_ROI_mask(points_local), points_global, np.nan)
            points_global = points_global[~np.isnan(points_global).any(axis=1)]
            points_local = np.where(create_ROI_mask(points_local), points_local, np.nan)
            points_local = points_local[~np.isnan(points_local).any(axis=1)]
            if len(points_local) < 6:
                continue

            # Downsampling
            if len(points_local) > max_num_points:
                n_step = int(len(points_local) / max_num_points)
                points_local = points_local[::n_step, :]
                points_global = points_global[::n_step, :]

            if line["Type"] == "Center":
                reference_line = {
                    "id": lane_id,
                    "lane_attribute": 7,
                    "passable_type": passable_type_mapping.get((road_id, lane_id), int(np.uint8(0b11111111))),
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
                result_map_dict["reference_lines"].append(reference_line)
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
                result_map_dict["lane_lines"].append(lane_line)

    return result_map_dict

def extract_cams(anno, timestamp, idx_anno, source_root, save_root):
    IMGS_FOLDER_NAME_MAPPING = {
        "CAM_FRONT": "rgb_front",
        "CAM_FRONT_LEFT": "rgb_front_left",
        "CAM_FRONT_RIGHT": "rgb_front_right",
        "CAM_BACK": "rgb_back",
        "CAM_BACK_LEFT": "rgb_back_left",
        "CAM_BACK_RIGHT": "rgb_back_right"
    }

    def extract_one_cam(cam_type: str):
        source_dir = os.path.join(
            source_root, "camera", IMGS_FOLDER_NAME_MAPPING[cam_type])
        image_file_list = glob(os.path.join(source_dir, "*.jpg"))
        image_file_list.sort()
        source_file = image_file_list[idx_anno]

        save_dir = os.path.join(
            save_root, "imgs", cam_type)
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, str(idx_anno) + ".jpg")

        shutil.copyfile(source_file, save_file)
        relative_path = "/".join(save_file.split("/")[-4:])

        sensor2ego_homo_trfs = np.array(anno["sensors"][cam_type]["cam2ego"])
        sensor2ego_translation = sensor2ego_homo_trfs[:-1, -1].tolist()
        sensor2ego_rotation = sensor2ego_homo_trfs[:-1, :-1].tolist()
        
        ego_box = find_ego_box(anno["bounding_boxes"])
        ego2global_translation = (-1 * np.array(ego_box["location"])).tolist()
        ego2global_rotation = R.from_euler("xyz", -1 * np.array(ego_box["rotation"]), degrees=True) \
                                .as_matrix().tolist()
        
        cam = {
            "data_path": relative_path,
            "type": cam_type,
            "timestamp": timestamp,
            "sensor2ego_translation": sensor2ego_translation,
            "sensor2ego_rotation": sensor2ego_rotation,
            "ego2global_translation": ego2global_translation,
            "ego2global_rotation": ego2global_rotation,
            "sensor2lidar_translation": sensor2ego_translation,
            "sensor2lidar_rotation": sensor2ego_rotation,
            "cam_intrinsic": anno["sensors"][cam_type]["intrinsic"]
        }
        return cam
    
    cams = {
        "CAM_FRONT": extract_one_cam("CAM_FRONT"),
        "CAM_FRONT_RIGHT": extract_one_cam("CAM_FRONT_RIGHT"),
        "CAM_FRONT_LEFT": extract_one_cam("CAM_FRONT_LEFT"),
        "CAM_BACK": extract_one_cam("CAM_BACK"),
        "CAM_BACK_LEFT": extract_one_cam("CAM_BACK_LEFT"),
        "CAM_BACK_RIGHT": extract_one_cam("CAM_BACK_RIGHT")
    }

    return cams

def extract_navi(anno):
    ROAD_OPTION_TO_COMMAND = {
        3: 0,
        1: 1,
        2: 2,
        -1: -1
    }

    speed_limit = 60 / 3.6 # default value
    for box in anno["bounding_boxes"]:
        if box["class"] == "traffic_sign":
            type_id_list = box["type_id"].split(".")
            if "speed_limit" in type_id_list:
                if box["affects_ego"]:
                    speed_limit = float(type_id_list[-1]) / 3.6

    navi = {
        "command": ROAD_OPTION_TO_COMMAND.get(anno["next_command"], 0),
        "intersection_distance": 999.9,
        "speed_limit": speed_limit
    }
    return navi

def extract_from_one_tar_file(tar_file: str, save_root: str):
    info_list = []
    
    # Read clip metadata
    folder = os.path.dirname(tar_file)
    scene = os.path.basename(tar_file).split(".")[0]
    scenario, map_name, route, weather = scene.split("_")

    # Extract .tar.gz
    if not os.path.isdir(os.path.join(folder, scene)):
        print(f"Extracting {tar_file}...")
        with tarfile.open(tar_file) as f:
            f.extractall(folder)
    
    # Prepare map
    map_dict = dict(np.load(
        os.path.join(MAP_ROOT, f"{map_name}_HD_map.npz"),
        allow_pickle=True)["arr"])

    # Anno
    anno_file_list = glob(os.path.join(folder, scene, "anno", "*.json.gz"))
    anno_file_list.sort()
    start_timestamp = time.time() * 1e3
    for idx_anno, anno_file in enumerate(tqdm(anno_file_list, desc=f"{scene}")):
    # for idx_anno, anno_file in enumerate(tqdm(anno_file_list[35:45])):
        with gzip.open(anno_file, 'rb') as f:
            anno = json.load(f)

        frame = get_default_frame(start_timestamp, idx_anno, len(anno_file_list))
        frame["can_bus"], (frame["ego2global_translation"], frame["ego2global_rotation"]) = extract_can_bus(anno)
        frame["agents"] = extract_agents(anno)
        frame["map"] = extract_maps(anno, map_dict)
        frame["cams"] = extract_cams(anno, frame["timestamp"], idx_anno,
            source_root=os.path.join(folder, scene),
            save_root=os.path.join(save_root, scene))
        frame["navi"] = extract_navi(anno)

        # pprint(frame)
        info_list.append(frame)
    return info_list

def work(tar_file, save_root):
    tar_file_name = os.path.basename(tar_file).split(".")[0]

    if tar_file_name in SKIP_LIST:
        return

    # if tar_file_name not in TEST_LIST:
    #     return
    
    if not(os.path.exists(os.path.join(save_root, tar_file_name))):
        os.makedirs(os.path.join(save_root, tar_file_name), exist_ok=True)
    
    if os.path.exists(os.path.join(save_root, tar_file_name, tar_file_name + ".pkl")):
        # print(f"{tar_file_name} already exists. Skipping...")
        return

    try:
        # print(f"Processing {tar_file_name}")
        info_list = extract_from_one_tar_file(tar_file, save_root)
        
        scene = {
            "metadata": {
                "version": "v0.0",
                "scene_id": str(hash(tar_file_name)),
                "desc": os.path.basename(tar_file).split(".")[0]
            },
            "infos": info_list
        }
        
        # save pickle
        with open(os.path.join(save_root, tar_file_name, tar_file_name + ".pkl"), "wb") as f:
            pickle.dump(scene, f)
        # print("Saved to ", os.path.join(save_root, tar_file_name, tar_file_name + ".pkl"))
        # # save json
        # with open(os.path.join(save_root, tar_file_name, tar_file_name + ".json"), "w", encoding="utf-8") as f:
        #     json.dump(scene, f, ensure_ascii=False, indent=4)
        # print("Saved to ", os.path.join(save_root, tar_file_name, tar_file_name + ".json"))
    except Exception as exc:
        print(exc)
        print("Failed to process ", tar_file_name)
        return

def prepare_maps(map_root):
    if os.path.exists(os.path.join(map_root, "map_cache.pkl")):
        print("Loading cached map: ", os.path.join(map_root, "map_cache.pkl"))
        with open(os.path.join(map_root, "map_cache.pkl"), "rb") as f:
            map_dict = pickle.load(f)
    else:
        map_file_list = glob(os.path.join(map_root, "*_HD_map.npz"))
        map_dict = {}
        for map_file in tqdm(map_file_list, desc="Preparing map(s)"):
            map_name = os.path.basename(map_file).split("_")[0]
            map_dict[map_name] = dict(np.load(map_file, allow_pickle=True)["arr"])
        
        with open(os.path.join(map_root, "map_cache.pkl"), "wb") as f:
            pickle.dump(map_dict, f)
        print("Saved to ", os.path.join(map_root, "map_cache.pkl"))
    return map_dict

def single_process(file_list: list):
    for file in tqdm(file_list, desc='Tar(s)'):
            work(file, SAVE_ROOT)

def multi_process(file_list: list):
    pbar = tqdm(total=len(file_list), desc='Tar(s)')
    pbar_update = lambda *args: pbar.update(1)

    pool = Pool(NUM_WORKER)
    for file in file_list:
        pool.apply_async(work, (file, SAVE_ROOT), callback=pbar_update)
    pool.close()
    pool.join()


if __name__ == "__main__":

    tar_file_list = glob(os.path.join(DATA_ROOT, "*.tar.gz"))
    tar_file_list.sort()
    if NUM_WORKER > 1:
        multi_process(tar_file_list)
    else:
        single_process(tar_file_list)

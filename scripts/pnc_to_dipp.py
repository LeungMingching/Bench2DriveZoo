import os
import pickle
import numpy as np

from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R

from pprint import pprint


TRAJECTORY_LENGTH = 6 # [s]

DATA_ROOT = "../data/Bench2Drive-PnC"
SAVE_ROOT = "../data/Bench2Drive-DIPP"
NUM_WORKER = 4

def extract_frame_info(metadata):
    frames = []
    if 'infos' not in metadata:
        print("Error: The key 'infos' is missing in metadata.")
        return frames

    for info in tqdm(metadata['infos'], desc="Frame(s)"):
        frame = {
            'timestamp': info.get('timestamp', -1),
            'navi':{
                'command': info.get('navi', {}).get('command', -1),  # 如果没有'navi'键，则返回默认值 -1
                'intersection_distance': info.get('navi', {}).get('intersection_distance', 0.0),
                'speed_limit': info.get('navi', {}).get('speed_limit', 0.0)},
                'turn_switch': info.get('turn_switch', -1),
                'ego_status': {
                    'position': info.get('can_bus', [0, 0])[:2],
                    'heading': info.get('can_bus', 0)[16],
                    'velocity': info.get('can_bus', [0, 0])[13:15],
                    'acceleration': info.get('can_bus', [0, 0])[7:9]
            },
            'agents': [],
            'reference_lines': [],
            'lane_lines': []
        }

        # 提取agents信息
        agents = info.get('agents', {})
        for agent_id, agent in agents.items():
            agent_info = {
                'id': agent.get('id',0),
                'pose': {
                    'BEV': {
                        'position': agent.get('pose', {}).get('ego', {}).get('position', [0, 0])[:2],
                        'heading': agent.get('pose', {}).get('ego', {}).get('heading', 0.0),
                        'velocity': agent.get('pose', {}).get('ego', {}).get('velocity', [0, 0])[:1],
                        'acceleration': agent.get('pose', {}).get('ego', {}).get('acceleration', [0, 0])[:1]
                    },
                    'UTM': {
                        'position': agent.get('pose', {}).get('odom', {}).get('position', [0, 0])[:2],
                        'heading': agent.get('pose', {}).get('odom', {}).get('heading', 0.0),
                        'velocity': agent.get('pose', {}).get('odom', {}).get('velocity', [0, 0])[:1],
                        'acceleration': agent.get('pose', {}).get('odom', {}).get('acceleration', [0, 0])[:1]
                    }
                },
                'type': agent.get('type', -1),
                'is_movable': agent.get('is_movable', False),
                'dimension': {
                    'corner_points': agent.get('dimension', {}).get('corner_points', []),
                    'width': agent.get('dimension', {}).get('width', 0.0),
                    'height': agent.get('dimension', {}).get('height', 0.0),
                    'length': agent.get('dimension', {}).get('length', 0.0)
                },
                'prediction': {
                    'lookahead_horizon': 0.0,
                    'mean_time_step': 0.0,
                    'future_poses': []
                }
            }
            frame['agents'].append(agent_info)

        # 提取参考线信息
        reference_lines = info.get('map', {}).get('reference_lines', [])
        for line in reference_lines:
            reference_line_info = {
                'lane_attribute': line.get('lane_attribute', []),
                'passable_type': line.get('passable_type', 0),
                'road_id': line.get('road_id', -1),
                'left_lane_line_id': line.get('left_lane_line_id', -1),
                'right_lane_line_id': line.get('right_lane_line_id', -1),
                'waypoint': {
                    'UTM': line.get('point', {}).get('odom', []),
                    'BEV': line.get('point', {}).get('ego', [])
                }
            }
            frame['reference_lines'].append(reference_line_info)

        # 提取车道线信息
        lane_lines = info.get('map', {}).get('lane_lines', [])
        for lane in lane_lines:
            lane_line_info = {
                'id': lane.get('id', -1),
                'type': lane.get('type', []),
                'separate_index': lane.get('separate_index', []),
                'waypoint': {
                    'UTM': lane.get('point', {}).get('odom', []),
                    'BEV': lane.get('point', {}).get('ego', [])
                }
            }
            frame['lane_lines'].append(lane_line_info)

        frames.append(frame)

    return frames

def extract_all_labels(frame_list):
    label_list = []

    for idx_current, frame in enumerate(frame_list):

        current_t = float(frame["timestamp"])
        traj_end_t = current_t + TRAJECTORY_LENGTH * 1000

        ego_translation = np.array(frame['ego_status']['position'])
        ego_heading = frame['ego_status']['heading']
        ego_rotation = R.from_euler('z', -ego_heading).as_matrix()[:2, :2]  # 计算逆旋转矩阵

        traj_pt_list = []
        for idx_nxt in range(idx_current, len(frame_list)):
            nxt_frame = frame_list[idx_nxt]
            
            if float(nxt_frame["timestamp"]) > traj_end_t:
                break

            position_global = np.array(nxt_frame['ego_status']['position'])
            velocity_global = np.array(nxt_frame['ego_status']['velocity'])
            acceleration_global = np.array(nxt_frame['ego_status']['acceleration'])
            heading_global = nxt_frame['ego_status']['heading']
            time = (float(nxt_frame['timestamp']) - current_t) / 1000.0

            relative_position = np.dot(ego_rotation, position_global[:2] - ego_translation[:2])

            relative_heading = heading_global - ego_heading

            rot_mat = R.from_euler('z', -relative_heading).as_matrix()[:2, :2]
            velocity_local = np.dot(rot_mat, velocity_global[:2])
            acceleration_local = np.dot(rot_mat, acceleration_global[:2])

            trajectory_point = [
                relative_position[0],
                relative_position[1],
                relative_heading,
                time,
                velocity_local[0],
                velocity_local[1],
                acceleration_local[0],
                acceleration_local[1]
            ]

            traj_pt_list.append(trajectory_point)

        # Fill out
        trajectory = {
            'lookahead_horizon': np.array(traj_pt_list)[-1, 4] if len(traj_pt_list) > 0 else 0,
            'mean_time_step': (np.array(traj_pt_list)[1:, 4] - np.array(traj_pt_list)[:-1, 4]).mean()  if len(traj_pt_list) > 1 else 0,
            'trajectory_points': traj_pt_list
        }

        label = {
            'timestamp': current_t,
            'clip_no': -1,  # 未使用，设为 -1
            'frame_idx': -1,  # 未使用，设为 -1
            'ilqr_trajectory': None,
            'rule_based_trajectory': None,
            'real_trajectory': trajectory
        }
        label_list.append(label)

    return label_list

def work(pkl_file):
    scene = os.path.basename(pkl_file).split(".")[0]

    # Make dirs
    observation_folder = os.path.join(SAVE_ROOT, scene, "observation")
    if not(os.path.exists(observation_folder)):
        os.makedirs(observation_folder, exist_ok=True)
    label_folder = os.path.join(SAVE_ROOT, scene, "label")
    if not(os.path.exists(label_folder)):
        os.makedirs(label_folder, exist_ok=True)
    
    # Convert
    with open(pkl_file, "rb") as f:
        metadata = pickle.load(f)
    frame_list = extract_frame_info(metadata)
    label_list = extract_all_labels(frame_list)

    # Save
    timestamp = frame_list[0]["timestamp"]
    with open(os.path.join(observation_folder, f"{timestamp}_frames.npy"), "wb") as f:
        np.save(f, frame_list)
    with open(os.path.join(label_folder, f"{timestamp}_label.npy"), "wb") as f:
        np.save(f, label_list)

def single_process(file_list: list):
    for file in tqdm(file_list, desc='Pkl(s)'):
        work(file)

def multi_process(file_list: list):
    pbar = tqdm(total=len(file_list), desc='Pkl(s)')
    pbar_update = lambda *args: pbar.update(1)

    pool = Pool(NUM_WORKER)
    for file in file_list:
        pool.apply_async(work, (file,), callback=pbar_update)
    pool.close()
    pool.join()


if __name__ == "__main__":

    pkl_file_list = glob(
        os.path.join(DATA_ROOT, "**/*.pkl"), recursive=True)
    if NUM_WORKER > 1:
        multi_process(pkl_file_list)
    else:
        single_process(pkl_file_list)
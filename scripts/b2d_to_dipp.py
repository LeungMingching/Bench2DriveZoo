import os
import tarfile
import gzip
import json

from glob import glob

from pprint import pprint

TSTEP = 0.1


def form_frame_template(timestamp: float, is_first=False, is_last=False):
    frame = {
        "prev": None if is_first else str(timestamp - TSTEP),
        "next": None if is_last else str(timestamp + TSTEP),
        "timestamp": str(timestamp),
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
    for idx_anno, anno_file in enumerate(anno_file_list[0:1]):
        with gzip.open(anno_file, 'rb') as f:
            anno = json.load(f)
        pprint(anno.keys())

    return


if __name__ == "__main__":

    DATA_ROOT = "../data/Bench2Drive-mini"

    tar_file_list = glob(os.path.join(DATA_ROOT, "*.tar.gz"))
    for tar_file in tar_file_list:
        extract_from_one_tar_file(tar_file)
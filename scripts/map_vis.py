# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @time      :2024-10-19 17:13
# @Author    :*******
# @FileName  :map_vis.py

import os
import json
import copy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from scipy.interpolate import interp1d
from PIL import Image
import random
import shutil
from scipy.spatial.transform import Rotation as Rot

vehicle_2_ground = 0.3

def readJson(jsonName):
    """
    """
    with open(jsonName, "r") as f0:
        data = json.load(f0)
    return data

def writeJsonToFile(path, jsonData):
    """
    """
    with open(path, "w") as f:
        f.write(json.dumps(jsonData,indent=4))

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

# proj ego->uv
def points_ego2img(pts_ego, extrinsics, intrinsics, vehicle_2_ground, axis_type):
    """
    """
    # import ipdb;ipdb.set_trace()
    if axis_type.upper() == "AES":
        cam_representation = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
    else:
        cam_representation = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = extrinsics @ pts_ego_4d.T  # x,y,z,1
    pts_cam_4d = cam_representation @ pts_cam_4d
    # 这里由于imu位置的问题，需要减去半个轮胎的高度
    # vehicle_2_ground = np.array([0, 0.315, 0, 0]).reshape((4, 1))
    # pts_cam_4d = pts_cam_4d + vehicle_2_ground

    uv = (intrinsics @ pts_cam_4d[:3, :]).T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.
    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):
        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        image = cv.circle(image, (int(x1), int(y1)), 1, (0,0,255), 4)
        image = cv.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv.LINE_AA)



def draw_polyline_ego_on_img(polyline_ego, img_bgr, extrinsics, intrinsics, color_bgr, thickness, vehicle_2_ground, save_name, axis_type="GQ", sample_dist=0.2):
    """
    """

    # assert polyline_ego.shape[1] == 3, "polyline must have z axis"
    # polyline_ego = interp_fixed_dist(line=LineString(polyline_ego), sample_dist=sample_dist)
    # import ipdb;ipdb.set_trace()
    uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics, vehicle_2_ground, axis_type)
    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    # print(save_name)
    # cv2.imwrite(save_name, img_bgr)
    # print("is_valid_points==>",is_valid_points)
    if is_valid_points.sum() == 0:
        return 

    tmp_list = []
    for i, valid in enumerate(is_valid_points):
        
        if valid:
            tmp_list.append(uv[i])
        else:
            if len(tmp_list) >= 2:
                tmp_vector = np.stack(tmp_list)
                tmp_vector = np.round(tmp_vector).astype(np.int32)
                draw_visible_polyline_cv2(
                    copy.deepcopy(tmp_vector),
                    valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
                    image=img_bgr,
                    color=color_bgr,
                    thickness_px=thickness,
                )
            tmp_list = []
    
    if len(tmp_list) >= 2:
        tmp_vector = np.stack(tmp_list)
        tmp_vector = np.round(tmp_vector).astype(np.int32)
        draw_visible_polyline_cv2(
            copy.deepcopy(tmp_vector),
            valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
            image=img_bgr,
            color=color_bgr,
            thickness_px=thickness,
        )

def odom2ego(frame, pts_odom):
    """
    """
    can_bus = frame['can_bus']
    ego_heading = can_bus[-2]
    # xyz
    ego2odom_translation = np.array(can_bus[0:3])
    # wxyz
    ego2odom_orientation = can_bus[3:7]
    quat = Rot.from_quat([ego2odom_orientation[1], ego2odom_orientation[2], ego2odom_orientation[3], ego2odom_orientation[0]])
    ego2odom_rt_matrix = np.eye(4)
    ego2odom_rt_matrix[:3, :3] = quat.as_matrix()
    ego2odom_rt_matrix[:3, 3] = ego2odom_translation
    odom2ego_rt_matrix = np.linalg.inv(ego2odom_rt_matrix)
    
    ego2odom_r_matrix = ego2odom_rt_matrix[:3, :3]
    odom2ego_r_matrix = np.linalg.inv(ego2odom_r_matrix[:3, :3])
    # print("pts_odom==>",pts_odom)
    new_points = []
    for pts in pts_odom:     
        new_points.append(np.dot(odom2ego_rt_matrix, np.append(pts, [1]))[:3].tolist())
    # print("new_points==>",new_points)

    return new_points


# def write2mp4(imagefolder,bevname, viewname, video_name):
    """
    """
    

    front = "center_120_camera"
    front_right = "right_front_camera"
    front_left = "left_front_camera"
    back = "back_camera"
    back_right = "right_back_camera"
    back_left = "left_back_camera"


    images = [img for img in os.listdir(os.path.join(imagefolder,front)) if img.endswith(".jpeg") or img.endswith(".png")]
    images.sort()
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps=5.0
    timeflag = images[0].split("_")[0]
    print("images[0]",images[0])
    img = cv.imread(os.path.join(os.path.join(imagefolder,front), images[0]))
    height,width,layers = img.shape
    video = cv.VideoWriter(video_name, fourcc, fps,(int(3*(width/3))+int(2*(height/3)), int(2*(height/3))))
    # video = cv.VideoWriter(video_name, fourcc, fps,(2640, 720))

    for image in tqdm(images):
        timestamp = image.split("_")[-1]
        img_front_right_name  = os.path.join(imagefolder,front_right + "/" + timeflag + "_right_front_camera_" + timestamp)
        img_front_left_name   = os.path.join(imagefolder,front_left + "/" + timeflag + "_left_front_camera_" + timestamp)
        img_back_name         = os.path.join(imagefolder,back + "/" + timeflag + "_back_camera_" + timestamp)
        img_back_right_name   = os.path.join(imagefolder,back_right + "/" + timeflag + "_right_back_camera_" + timestamp)
        img_back_left_name    = os.path.join(imagefolder,back_left + "/" + timeflag + "_left_back_camera_" + timestamp)
        img_bev = os.path.join(bevname,timestamp)
        # print("img_front_right_name==>",img_front_right_name)
        # print("img_bev==>",img_bev)

        
        if (os.path.exists(os.path.join(imagefolder,front)+"/"+image)
            and os.path.exists(img_front_right_name)
            and os.path.exists(img_front_left_name)
            and os.path.exists(img_back_name)
            and os.path.exists(img_back_right_name)
            and os.path.exists(img_back_left_name)
            and os.path.exists(img_bev)):
            
            img_front       = cv.imread(os.path.join(imagefolder,front)+"/"+image)
            img_front = cv.resize(img_front, (int(1920/3), int(1080/3)), interpolation = cv.INTER_CUBIC)
            img_front_right = cv.imread(img_front_right_name)
            img_front_right = cv.resize(img_front_right, (int(1920/3), int(1080/3)), interpolation = cv.INTER_CUBIC)
            img_front_left  = cv.imread(img_front_left_name)
            img_front_left = cv.resize(img_front_left, (int(1920/3), int(1080/3)), interpolation = cv.INTER_CUBIC)
            img_bev  = cv.imread(img_bev)
            

            img_back        = cv.imread(img_back_name)
            img_back = cv.resize(img_back, (int(1920/3), int(1080/3)), interpolation = cv.INTER_CUBIC)
            img_back_right  = cv.imread(img_back_right_name)
            img_back_right = cv.resize(img_back_right, (int(1920/3), int(1080/3)), interpolation = cv.INTER_CUBIC)
            img_back_left   = cv.imread(img_back_left_name)
            img_back_left = cv.resize(img_back_left, (int(1920/3), int(1080/3)), interpolation = cv.INTER_CUBIC)


            img1 = np.concatenate((img_front_left, img_front), axis=1)
            img2 = np.concatenate((img1, img_front_right), axis=1)
            img3 = np.concatenate((img_back_left, img_back), axis=1)
            img4 = np.concatenate((img3, img_back_right), axis=1)

            img_res = np.concatenate((img2, img4), axis=0)
            h,w = img_res.shape[:2]

            img_bev = cv.resize(img_bev, (int(h/600*600),h), interpolation = cv.INTER_CUBIC)
            img_res = np.concatenate((img_res, img_bev), axis=1)
            

            save_name = os.path.join(viewname,timestamp)
            cv.imwrite(save_name,img_res)
            video.write(img_res)
    video.release()

    print(f"video has saved to{video_name}")

# def getextrinsics(R, T):
    """
    """
    extrinsics_martix = np.eye(4)
    extrinsics_martix[:3,:3] = R
    extrinsics_martix[:3,3] = T

    return extrinsics_martix

def drawmemmap(frame, elements, img, extrinsics, intrinsics, color):
    """
    """
    if frame["map"][elements]:
        for element in frame["map"][elements]:
            pts_ego = element["point"]["ego"]
            if pts_ego:
                draw_polyline_ego_on_img(pts_ego, img, extrinsics, intrinsics, color, 2, 0, "img.png", axis_type="GQ", sample_dist=0.2)

def drawframes(data,i,img,extrinsics, intrinsics):
    """
    """
    frame = data["infos"][i]
    drawmemmap(frame, "lane_lines", img, extrinsics, intrinsics, [255,0,0])
    drawmemmap(frame, "reference_lines", img, extrinsics, intrinsics, [0,255,0])
    drawmemmap(frame, "stop_lines", img, extrinsics, intrinsics, [0,0,255])
    drawmemmap(frame, "curbs", img, extrinsics, intrinsics, [0,255,255])

    return img


def drawagents(poly_lines, img, color):
    """
    """
    for poly_line in poly_lines:
        pts_ego = poly_line["point"]["ego"]
        for i in range(len(pts_ego) - 1):
            x1 = pts_ego[i][0]
            y1 = pts_ego[i][1]
            x2 = pts_ego[i + 1][0]
            y2 = pts_ego[i + 1][1]
            # img = cv.circle(img, (int((-1*y1+50)*4)+100, int((-1*x1+100)*4)), 1, (0,0,255), 1)
            img = cv.line(img, (int((-1*y1+50)*4)+100, int((-1*x1+100)*4)), (int((-1*y2+50)*4)+100, int((-1*x2+100)*4)), color, thickness=1, lineType=cv.LINE_AA)

def seektracking(data,i,trackid):
    """
    """
    former = []
    later = []
    timestamp1 = data["infos"][i]["timestamp"]
    for f in range(i,0,-1):
        aginfo = data["infos"][f]["agents"]
        # print("aginfo",aginfo)
        timestamp0 = data["infos"][f]["timestamp"]
        if trackid in aginfo :
            former.insert(0,f)
        if float(timestamp1) - float(timestamp0) >5000:
            break
    for f in range(i,len(data["infos"])):
        aginfo = data["infos"][f]["agents"]
        timestamp2 = data["infos"][f]["timestamp"]
        if trackid in aginfo :
            later.append(f)
        if float(timestamp2) - float(timestamp1) >5000:
            break
    return former, later

def plotagent(data,index):
    """
    """
    FRAME = data["infos"][index]
    agentsinfo = data["infos"][index]["agents"]
    mapinfo = data["infos"][index]["map"]
    height = 600
    width = 600
    img = np.ones((height, width, 3),dtype = np.uint8)
    img*=255
    img = cv.circle(img, (300,400), 1, (0,255,0), 4)
    cv.putText(img,"ego",(300,400),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),1)
    if mapinfo:
        
        if mapinfo["lane_lines"]:
            poly_lines = mapinfo["lane_lines"]
            drawagents(poly_lines, img, (147,20,255))

        if mapinfo["reference_lines"]:
            poly_lines = mapinfo["reference_lines"]
            drawagents(poly_lines, img, (0,255,0))
            
        if mapinfo["stop_lines"]:
            poly_lines = mapinfo["stop_lines"]
            drawagents(poly_lines, img, (0,0,255))
            
        if mapinfo["curbs"]:
            poly_lines = mapinfo["curbs"]
            drawagents(poly_lines, img, (0,255,255))

    for k,v in agentsinfo.items():
        idx = k 
        egopoint = v["pose"]["ego"]["position"]
        cornerpoints = v["dimension"]["corner_points"]
        cornerpoints.append(cornerpoints[0])

        x,y,z = egopoint
        if -50<x<100 and -50<y<50:
            img = cv.circle(img, (int((-1*y+50)*4)+100, int((-1*x+100)*4)), 1, (0,0,255), 6)
            # cv.putText(img,idx,(int((-1*y+50)*4)+100, int((-1*x+100)*4)),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),1)
        
        for i in range(len(cornerpoints) - 1):
            x1 = cornerpoints[i][0]
            y1 = cornerpoints[i][1]
            x2 = cornerpoints[i + 1][0]
            y2 = cornerpoints[i + 1][1]
            img = cv.line(img, (int((-1*y1+50)*4)+100, int((-1*x1+100)*4)), (int((-1*y2+50)*4)+100, int((-1*x2+100)*4)), (0,0,255), thickness=1, lineType=cv.LINE_AA)

        former,later = seektracking(data,index,k)

        egopoint_fomer = []
        for j in former:
            egopoint = data["infos"][j]["agents"][k]["pose"]["odom"]["position"]
            # x,y,z = egopoint
            # img = cv.circle(img, (int((-1*y+50)*4)+100, int((-1*x+100)*4)), 1, (144,32,208), 1)
            egopoint_fomer.append(egopoint)
        egopoint_fomer = odom2ego(FRAME, egopoint_fomer)
        for i in range(len(egopoint_fomer)-1):
            x1 = egopoint_fomer[i][0]
            y1 = egopoint_fomer[i][1]
            x2 = egopoint_fomer[i + 1][0]
            y2 = egopoint_fomer[i + 1][1]
            img = cv.line(img, (int((-1*y1+50)*4)+100, int((-1*x1+100)*4)), (int((-1*y2+50)*4)+100, int((-1*x2+100)*4)), (144,32,208), thickness=1, lineType=cv.LINE_AA)

            # x,y,z = egopoint
            # if -50<x<100 and -50<y<50:

            #     img = cv.circle(img, (int((-1*y+50)*4)+100, int((-1*x+100)*4)), 1, (255,0,0), 4)
        egopoint_later = []
        for n in later:
            egopoint = data["infos"][n]["agents"][k]["pose"]["odom"]["position"]
            # x,y,z = egopoint
            # img = cv.circle(img, (int((-1*y+50)*4)+100, int((-1*x+100)*4)), 1, (139,104,0), 1)
            egopoint_later.append(egopoint)

        egopoint_later = odom2ego(FRAME, egopoint_later)
        for i in range(len(egopoint_later)-1):
            x1 = egopoint_later[i][0]
            y1 = egopoint_later[i][1]
            x2 = egopoint_later[i + 1][0]
            y2 = egopoint_later[i + 1][1]
            img = cv.line(img, (int((-1*y1+50)*4)+100, int((-1*x1+100)*4)), (int((-1*y2+50)*4)+100, int((-1*x2+100)*4)), (139,104,0), thickness=1, lineType=cv.LINE_AA)
            # x,y,z = egopoint
            # if -50<x<100 and -50<y<50:
            #     img = cv.circle(img, (int((-1*y+50)*4)+100, int((-1*x+100)*4)), 1, (0,255,0), 4)



    return img 

def mkdir(path,outpath,tempdir,viewdir,bevdir):
    """
    """

    if not(os.path.exists(outpath)):
        os.mkdir(outpath)
    if not(os.path.exists(tempdir)):
        os.mkdir(tempdir)
    if not(os.path.exists(viewdir)):
        os.mkdir(viewdir)
    if not(os.path.exists(bevdir)):
        os.mkdir(bevdir)

    listpath = os.listdir(path)
    for each in listpath:
        savesample = os.path.join(outpath, each)
        savetemp = os.path.join(tempdir, each)
        if not(os.path.exists(savesample)):
            os.mkdir(savesample)
        if not(os.path.exists(savetemp)):
            os.mkdir(savetemp)

def sample(data, samplepath, percent):
    """
    """

    N = [i for i in range(len(data["infos"]))]
    m = int(len(N)*percent)
    samples = random.sample(N,m)

    for i in range(len(data["infos"])):
        if i in samples:
            data_path_CAM_FRONT       = "../"+data["infos"][i]["cams"]["CAM_FRONT"]["data_path"]
            data_path_CAM_FRONT_RIGHT = "../"+data["infos"][i]["cams"]["CAM_FRONT_RIGHT"]["data_path"]
            data_path_CAM_FRONT_LEFT  = "../"+data["infos"][i]["cams"]["CAM_FRONT_LEFT"]["data_path"]
            data_path_CAM_BACK        = "../"+data["infos"][i]["cams"]["CAM_BACK"]["data_path"]
            data_path_CAM_BACK_RIGHT  = "../"+data["infos"][i]["cams"]["CAM_BACK_RIGHT"]["data_path"]
            data_path_CAM_BACK_LEFT   = "../"+data["infos"][i]["cams"]["CAM_BACK_LEFT"]["data_path"]

            shutil.copy2(data_path_CAM_FRONT, os.path.join(samplepath, "/".join(data_path_CAM_FRONT.split("/")[-2:])))
            shutil.copy2(data_path_CAM_FRONT_RIGHT, os.path.join(samplepath, "/".join(data_path_CAM_FRONT_RIGHT.split("/")[-2:])))
            shutil.copy2(data_path_CAM_FRONT_LEFT, os.path.join(samplepath, "/".join(data_path_CAM_FRONT_LEFT.split("/")[-2:])))
            shutil.copy2(data_path_CAM_BACK, os.path.join(samplepath, "/".join(data_path_CAM_BACK.split("/")[-2:])))
            shutil.copy2(data_path_CAM_BACK_RIGHT, os.path.join(samplepath, "/".join(data_path_CAM_BACK_RIGHT.split("/")[-2:])))
            shutil.copy2(data_path_CAM_BACK_LEFT, os.path.join(samplepath, "/".join(data_path_CAM_BACK_LEFT.split("/")[-2:])))
    

def main():
    """
    """

    file = "1732093251656"

    ########################################################################################
    jsonpath = f"../data/Bench2Drive-DIPP/{file}/{file}.json"
    imagepath = f"../data/Bench2Drive-DIPP/{file}/imgs"
    samplepath = f"../data/Bench2Drive-DIPP/{file}/sample"
    tempdir = f"../data/Bench2Drive-DIPP/{file}/temp"
    viewdir = f"../data/Bench2Drive-DIPP/{file}/res"
    bevdir = f"../data/Bench2Drive-DIPP/{file}/bev"
    videodirtosave = f"../data/Bench2Drive-DIPP/{file}/out.mp4"
    sampleFlag = False
    percent = 0.05

    ########################################################################################
    # mkdir
    mkdir(imagepath,samplepath,tempdir,viewdir,bevdir)
    ########################################################################################
    data = readJson(jsonpath)

    if sampleFlag:
        sample(data, samplepath, percent)

    flag=False
    
    for i in tqdm(range(len(data["infos"]))):

        agents = data["infos"][i]["agents"]
        mapinfo = data["infos"][i]["map"]
        # print(data["infos"][i]["cams"]["CAM_FRONT"]["data_path"])


        # data_path_CAM_FRONT       = "./sample/"+("/".join(data["infos"][i]["cams"]["CAM_FRONT"]["data_path"].split("/")[2:]))
        # data_path_CAM_FRONT_RIGHT = "./sample/"+("/".join(data["infos"][i]["cams"]["CAM_FRONT_RIGHT"]["data_path"].split("/")[2:]))
        # data_path_CAM_FRONT_LEFT  = "./sample/"+("/".join(data["infos"][i]["cams"]["CAM_FRONT_LEFT"]["data_path"].split("/")[2:]))
        # data_path_CAM_BACK        = "./sample/"+("/".join(data["infos"][i]["cams"]["CAM_BACK"]["data_path"].split("/")[2:]))
        # data_path_CAM_BACK_RIGHT  = "./sample/"+("/".join(data["infos"][i]["cams"]["CAM_BACK_RIGHT"]["data_path"].split("/")[2:]))
        # data_path_CAM_BACK_LEFT   = "./sample/"+("/".join(data["infos"][i]["cams"]["CAM_BACK_LEFT"]["data_path"].split("/")[2:]))

        # if (os.path.exists(data_path_CAM_FRONT)
        #     and os.path.exists(data_path_CAM_FRONT_RIGHT)
        #     and os.path.exists(data_path_CAM_FRONT_LEFT)
        #     and os.path.exists(data_path_CAM_BACK)
        #     and os.path.exists(data_path_CAM_BACK_RIGHT)
        #     and os.path.exists(data_path_CAM_BACK_LEFT)):

        #     img_front = cv.imread(data_path_CAM_FRONT)
        #     img_front_right = cv.imread(data_path_CAM_FRONT_RIGHT)
        #     img_front_left = cv.imread(data_path_CAM_FRONT_LEFT)
        #     img_back = cv.imread(data_path_CAM_BACK)
        #     img_back_right = cv.imread(data_path_CAM_BACK_RIGHT)
        #     img_back_left = cv.imread(data_path_CAM_BACK_LEFT)

        #     if data["infos"][i]["map"]:

        #         R_front       = np.array(data["infos"][i]["cams"]["CAM_FRONT"]["lidar2sensor_rotation"])
        #         T_front       = np.array(data["infos"][i]["cams"]["CAM_FRONT"]["lidar2sensor_translation"])
        #         R_front_right = np.array(data["infos"][i]["cams"]["CAM_FRONT_RIGHT"]["lidar2sensor_rotation"])
        #         T_front_right = np.array(data["infos"][i]["cams"]["CAM_FRONT_RIGHT"]["lidar2sensor_translation"])
        #         R_front_left  = np.array(data["infos"][i]["cams"]["CAM_FRONT_LEFT"]["lidar2sensor_rotation"])
        #         T_front_left  = np.array(data["infos"][i]["cams"]["CAM_FRONT_LEFT"]["lidar2sensor_translation"])
        #         R_back        = np.array(data["infos"][i]["cams"]["CAM_BACK"]["lidar2sensor_rotation"])
        #         T_back        = np.array(data["infos"][i]["cams"]["CAM_BACK"]["lidar2sensor_translation"])
        #         R_back_right  = np.array(data["infos"][i]["cams"]["CAM_BACK_RIGHT"]["lidar2sensor_rotation"])
        #         T_back_right  = np.array(data["infos"][i]["cams"]["CAM_BACK_RIGHT"]["lidar2sensor_translation"])
        #         R_back_left   = np.array(data["infos"][i]["cams"]["CAM_BACK_LEFT"]["lidar2sensor_rotation"])
        #         T_back_left   = np.array(data["infos"][i]["cams"]["CAM_BACK_LEFT"]["lidar2sensor_translation"])

        #         extrinsics_front       = getextrinsics(R_front, T_front)
        #         extrinsics_front_right = getextrinsics(R_front_right, T_front_right)
        #         extrinsics_front_left  = getextrinsics(R_front_left, T_front_left)
        #         extrinsics_back        = getextrinsics(R_back, T_back)
        #         extrinsics_back_right  = getextrinsics(R_back_right, T_back_right)
        #         extrinsics_back_left   = getextrinsics(R_back_left, T_back_left)

        #         intrinsics_front       = data["infos"][i]["cams"]["CAM_FRONT"]['cam_intrinsic']
        #         # import pdb; pdb.set_trace()

        #         intrinsics_front_right = data["infos"][i]["cams"]["CAM_FRONT_RIGHT"]['cam_intrinsic']
        #         intrinsics_front_left  = data["infos"][i]["cams"]["CAM_FRONT_LEFT"]['cam_intrinsic']
        #         intrinsics_back        = data["infos"][i]["cams"]["CAM_BACK"]['cam_intrinsic']
        #         intrinsics_back_right  = data["infos"][i]["cams"]["CAM_BACK_RIGHT"]['cam_intrinsic']
        #         intrinsics_back_left   = data["infos"][i]["cams"]["CAM_BACK_LEFT"]['cam_intrinsic']
            
        #         img_front = drawframes(data,i,img_front, extrinsics_front, intrinsics_front)
        #         img_front_right = drawframes(data,i,img_front_right, extrinsics_front_right, intrinsics_front_right)               
        #         img_front_left = drawframes(data,i,img_front_left, extrinsics_front_left, intrinsics_front_left)      
        #         img_back = drawframes(data,i,img_back, extrinsics_back, intrinsics_back)           
        #         img_back_right = drawframes(data,i,img_back_right, extrinsics_back_right, intrinsics_back_right)            
        #         img_back_left = drawframes(data,i,img_back_left, extrinsics_back_left, intrinsics_back_left)
                
        #     save_name_front = data_path_CAM_FRONT.replace(samplepath.split("/")[-1],tempdir.split("/")[-1])
        #     cv.imwrite(save_name_front,img_front)
        #     save_name_front_right = data_path_CAM_FRONT_RIGHT.replace(samplepath.split("/")[-1],tempdir.split("/")[-1])
        #     cv.imwrite(save_name_front_right,img_front_right)
        #     save_name_front_left = data_path_CAM_FRONT_LEFT.replace(samplepath.split("/")[-1],tempdir.split("/")[-1])
        #     cv.imwrite(save_name_front_left,img_front_left)
        #     save_name_back = data_path_CAM_BACK.replace(samplepath.split("/")[-1],tempdir.split("/")[-1])
        #     cv.imwrite(save_name_back,img_back)
        #     save_name_back_right = data_path_CAM_BACK_RIGHT.replace(samplepath.split("/")[-1],tempdir.split("/")[-1])
        #     cv.imwrite(save_name_back_right,img_back_right)
        #     save_name_back_left = data_path_CAM_BACK_LEFT.replace(samplepath.split("/")[-1],tempdir.split("/")[-1])
        #     cv.imwrite(save_name_back_left,img_back_left)


        img_bev = plotagent(data, i)
        save_name_bev = os.path.join(bevdir, f"{i}.png")
        cv.imwrite(save_name_bev,img_bev)

    # write2mp4(tempdir, bevdir, viewdir, videodirtosave)
    



if __name__ == "__main__":
    main()









#!/usr/bin/env python3

"""
Real-time grasp detection.
"""

import argparse
from pathlib import Path
import time

# import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg
import torch

from grasp_lib.msg import Grasp as Gr
from vgn import vis
from vgn.detection import *
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform


class GraspDetectionServer(object):
    def __init__(self, model_path):
        # define frames
        self.task_frame_id = "task"
        self.cam_frame_id = "ptu_camera_depth_optical_frame"
        self.grasp_pub = rospy.Publisher("vgn/output", Gr, queue_size=1)
        self.T_cam_task = Transform(
            Rotation.from_quat([0.960, 0.000, -0.000, 0.282]), [-0.131, 0.292, 0.825])

        # define camera parameters
        self.cam_topic_name = "/ptu_camera/camera/aligned_depth_to_color/image_raw"

        cam_info = rospy.wait_for_message('ptu_camera/camera/color/camera_info', sensor_msgs.msg.CameraInfo, timeout=rospy.Duration(1))
        self.intrinsic = CameraIntrinsic(cam_info.width, cam_info.height, cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5])

        # construct the grasp planner object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)

        # initialize the visualization
        vis.clear()
        vis.draw_workspace(0.3)

        # subscribe to the camera
        self.img = None
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

        # setup cb to detect grasps
        rospy.Timer(rospy.Duration(0.1), self.detect_grasps)

    def sensor_cb(self, msg):
        depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
        self.img = depth_image.astype(np.float32) * 0.001

    def grasp_paser(self, grasps, scores):
        vis_grasp = Gr()

        if grasps:
            index = np.argmax(scores)
            # print("translation: ",grasps[index].pose.translation)
            # print("rotation: ",grasps[index].pose.rotation.as_quat())
            # print("width: ",grasps[index].width)
            # print("Scores: ",scores[index])

            vis_grasp.name = "vgn"
            vis_grasp.pose.position.x = grasps[index].pose.translation[0]
            vis_grasp.pose.position.y = grasps[index].pose.translation[1]
            vis_grasp.pose.position.z = grasps[index].pose.translation[2]
            rotation = grasps[index].pose.rotation.as_quat()
            vis_grasp.pose.orientation.x = rotation[0]
            vis_grasp.pose.orientation.y = rotation[1]
            vis_grasp.pose.orientation.z = rotation[2]
            vis_grasp.pose.orientation.w = rotation[3]
            vis_grasp.width_meter = grasps[index].width
            vis_grasp.quality = scores[index]
            self.grasp_pub.publish(vis_grasp)

    def detect_grasps(self, _):
        if self.img is None:
            return

        tic = time.time()
        self.tsdf = TSDFVolume(1.0, 40)
        self.tsdf.integrate(self.img, self.intrinsic, self.T_cam_task)
        print("Construct tsdf ", time.time() - tic)

        tic = time.time()
        tsdf_vol = self.tsdf.get_grid()
        voxel_size = self.tsdf.voxel_size
        print("Extract grid  ", time.time() - tic)

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        print("Forward pass   ", time.time() - tic)

        tic = time.time()
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        print("Filter         ", time.time() - tic)

        vis.draw_quality(qual_vol, voxel_size, threshold=0.01)

        tic = time.time()
        grasps, scores = select(qual_vol, rot_vol, width_vol, 0.90, 1)
        num_grasps = len(grasps)
        if num_grasps > 0:
            idx = np.random.choice(num_grasps, size=min(5, num_grasps), replace=False)
            grasps, scores = np.array(grasps)[idx], np.array(scores)[idx]
        grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps]
        print("Select grasps  ", time.time() - tic)

        vis.clear_grasps()
        rospy.sleep(0.01)
        tic = time.time()
        self.grasp_paser(grasps,scores)
        vis.draw_grasps(grasps, scores, 0.05)
        print("Visualize      ", time.time() - tic)

        self.img = None
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args, unknown = parser.parse_known_args()

    rospy.init_node("panda_detection")
    GraspDetectionServer(args.model)
    rospy.spin()

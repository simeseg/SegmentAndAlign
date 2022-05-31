# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:20:15 2022

@author: IntekPlus
"""

import numpy as np
import open3d as o3d
import utils
from icp_GN import icp_GN
from PCSegmentAndAlign import segmentAndAlign

#read model 
model = o3d.io.read_point_cloud("./BoltData/bolt_model.pcd") 

#read scene
root_dir = "D:/PointCloudSegmentation/Dataset/source/results/test"
sa = segmentAndAlign(root_dir, 1)
sa.project(sa.scene_points, sa.rgb, sa.camera_pose)
scene, _ = sa.clip_bolt_pcd(0)
#scene = o3d.io.read_point_cloud("./BoltData/boltpcd2.pcd")

#model = utils.NanInfScale(model, 1000)
#scene = utils.NanInfScale(scene, 1000)

#model.points = o3d.utility.Vector3dVector(model.points - np.mean(model.points, axis=0)) 
#scene.points = o3d.utility.Vector3dVector(scene.points - np.mean(scene.points, axis=0))
#scene.points = o3d.utility.Vector3dVector(utils.Rot3D(np.asarray(scene.points), 0, 0, 0, 0, 0, 0))

#args: grid_size, distance threshold, iterations, error mode, model, scene
icp = icp_GN(1, 1, 10, "plane", False, model, scene)
H = icp.get_final_transformation()
print("H")
print(H)
#icp = icp_LM(1, 1, 20, "plane", True, model, scene)

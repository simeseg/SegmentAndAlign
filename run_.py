# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:20:15 2022

@author: IntekPlus
"""


from eigenvalue_init import align
from segmentPCD import segmentPCD

#read model 
model_dir = "../BoltData/bolt_model.pcd"

#read scene
scene_root_dir = "../../Dataset/source/results/test"

seg = segmentPCD(model_dir, scene_root_dir, 4)

#args: grid_size, distance threshold, iterations, error mode, viz, model, scene
align = align(1, 5, 2000, "point", seg.model, seg.clip_bolt_pcd(0))


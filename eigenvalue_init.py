# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:20:15 2022

@author: IntekPlus
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import utils
import copy


class align():
    def __init__(self, grid_size, distance_threshold, iterations, mode, model, scene, full_scene):
        
        self.dist = distance_threshold
        self.mode = mode
        self.iterations = iterations
        
        #shift the cloud by the mean
        self.model, self.staticMean = self.shift_by_mean(model)
        self.scene, self.dynamicMean = self.shift_by_mean(scene)
        
        #scale pointclouds
        self.allpts = np.concatenate((self.model.points, self.scene.points), axis=0)
        self.min, self.max = np.min(self.allpts, axis=0), np.max(self.allpts, axis=0)
        self.scale = 1 #/np.max(self.max - self.min)
        
        # get model pcd downsample and tree and normals
        self.norm_param = o3d.geometry.KDTreeSearchParamKNN(50)
        self.model.estimate_normals(self.norm_param)
        self.model.orient_normals_consistent_tangent_plane(k=50)
        self.model = self.model.voxel_down_sample(grid_size)
        self.normals = np.asarray(self.model.normals)
        self.tree = o3d.geometry.KDTreeFlann(self.model)
        
        # get scene downsampled
        self.scene = self.scene.voxel_down_sample(grid_size)
        self.full_scene = full_scene
        
        self.alignedCloud_left  = o3d.geometry.PointCloud()
        self.alignedCloud_right = o3d.geometry.PointCloud()
        
        R, R_eta = self.eigenvalue_init()
        
        self.alignment_left(R)
        self.alignment_right(R_eta)
        
        self.alignedCloud_left
        self.alignedCloud_right
        
        self.loss = o3d.pipelines.registration.CauchyLoss(k=0.5)
        
        if self.mode == 'Point' or 'point':
            self.register_mode = o3d.pipelines.registration.TransformationEstimationPointToPoint()     
        elif self.mode == 'Plane' or 'plane':
            self.register_mode = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            
            
        self.H, self.rmse, self.fitness = self.registration(self.alignedCloud_left)
        self.H_eta, self.rmse_eta, self.fitness_eta = self.registration(self.alignedCloud_right)  
        
        self.H_out, self.R_out, self.alignedCloud, self.fitness_out, self.rmse_out = (self.H, R, self.alignedCloud_left, self.fitness, self.rmse) if (self.rmse < self.rmse_eta) else (self.H_eta, R_eta, self.alignedCloud_right, self.fitness_eta, self.rmse_eta) 
        
        #self.draw_registration_result(self.alignedCloud_left, self.model, self.H)
        #self.draw_registration_result(self.alignedCloud_right, self.model, self.H_eta)
        
        
        self.h1 = np.eye(4)
        self.h1[:3,3] = -self.dynamicMean
        self.h4 = np.eye(4)
        self.h4[:3,3] = self.staticMean
        self.h2 = np.eye(4)
        self.h2[:3,:3] = self.R_out
        self.h_final = self.h4@self.H_out@self.h2@self.h1
        print(" inlier rmse: ", self.rmse)
        print(" fitness: ", self.fitness_out)
        print(" transform: ", self.h_final)
        
        self.draw_registration_result(self.alignedCloud, self.model, self.H_out)
        self.draw_registration_result(self.model, self.full_scene, np.linalg.inv(self.h_final))
        
    def alignment_left(self, R):
        #initialize start using eigenvalue decomp
        points = R@(np.asarray(self.scene.points)).transpose(1,0) 
        self.alignedCloud_left.points = o3d.utility.Vector3dVector(np.asarray(points.T))
        
    def alignment_right(self, R):
        #initialize start using eigenvalue decomp
        points = R@(np.asarray(self.scene.points)).transpose(1,0) 
        self.alignedCloud_right.points = o3d.utility.Vector3dVector(np.asarray(points.T))
                                                                    
    def shift_by_mean(self, cloud):
        points = np.asarray(cloud.points)
        mean = np.mean(points, axis=0)
        cloud_out = o3d.geometry.PointCloud()
        cloud_out.points = o3d.utility.Vector3dVector(points - mean)
        return cloud_out, mean

    def eigenvalue_init(self):
        
        #model eigenvectors
        model_points = np.asarray(self.model.points).T
        cov_model = model_points@(model_points.T)
        eig_model, v_model = np.linalg.eig(cov_model)
        v1 = v_model[:, np.argmax(eig_model)]
        v1 /= np.linalg.norm(v1)
        
        #model eigenvectors
        scene_points = np.asarray(self.scene.points).T
        cov_scene = scene_points@(scene_points.T)
        eig_scene, v_scene = np.linalg.eig(cov_scene)
        v2 = v_scene[:, np.argmax(eig_scene)]
        v2 /= np.linalg.norm(v2)
        
        #rotation axis and matrix
        theta = np.arccos(min(v1.dot(v2), 1))
        ax = -np.cross(v1,v2)
        if np.linalg.norm(ax) != 0:
            ax /= np.linalg.norm(ax)
        #skew symmetric form
        s_ax = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(theta)*s_ax + (1- np.cos(theta))*s_ax@s_ax
        
        #rotation axis and matrix for the flipped pcd
        eta = theta + np.pi
        R_eta = np.eye(3) + np.sin(eta)*s_ax + (1- np.cos(eta))*s_ax@s_ax
    
        #print(np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(R_eta[:,1]), theta*180/np.pi, ax)
        return R, R_eta


    def registration(self, cloud_in):
        
        transform = o3d.pipelines.registration.registration_icp(self.model, cloud_in, self.dist, np.eye(4),
        self.register_mode, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.iterations))
        
        return transform.transformation, transform.inlier_rmse, transform.fitness
    
    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0.000])
        #target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        #target_temp.paint_uniform_color([ 0, 0, 1])
        #source_temp.paint_uniform_color([ 0, 1, 0])
        #source.paint_uniform_color([ 1, 0, 0])
        
        o3d.visualization.draw_geometries([source_temp, target_temp])
                                          #zoom=0.4459,
                                          #front=[0.9288, -0.2951, -0.2242],
                                          #lookat=[1.6784, 2.0612, 1.4451],
                                          #up=[-0.3402, -0.9189, -0.1996])
        
    
    
    
    
    
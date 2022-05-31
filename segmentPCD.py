# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:49:23 2022

@author: IntekPlus
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class segmentPCD():
    def __init__(self, model_dir, scene_root_dir, scene_id):
        
        #orthogonal
        self.PrincipalPointOffsetX=1330.4902
        self.PrincipalPointOffsetY=1040.2366
        self.FocalLengthX=3372.0126
        self.FocalLengthY=3367.3384
        self.SkewCoefficient=0.00000000
        self.RadialDistortion1=-0.0824
        self.RadialDistortion2=-0.2857
        self.RadialDistortion3=-0.7985
        self.TangentialDistortion1=-0.0004
        self.TangentialDistortion2=0.0009

        
        self.K = np.array([self.FocalLengthX, self.SkewCoefficient, self.PrincipalPointOffsetX, 0, self.FocalLengthY, self.PrincipalPointOffsetY, 0, 0, 1]).reshape(3,3)
        #self.camera_pose = np.array([0, -5, 0, -18, -5, 800]) #extrinsic
        self.camera_pose = np.array([-0.993, 0.007, 0.113, 2.784, -0.008, -0.999, -0.009, 13.766, 0.113, -0.010, 0.993, 1125.17, 0, 0, 0, 1]).reshape(4, 4)
        
        self.width = 2448
        self.height = 2048
        
        #set scene depth projection
        self.Zpos = np.zeros([self.height, self.width])
        self.Xpos = np.zeros([self.height, self.width])
        self.Ypos = np.zeros([self.height, self.width])
        
        self.root_dir = scene_root_dir
        self.model_dir = model_dir
        
        #get bolt PLY file
        self.model = o3d.io.read_point_cloud(self.model_dir)
        
        #get RGB
        self.scene_id = scene_id
        self.rgb = plt.imread(self.root_dir + "/%.1d/fast_rcnn_result.jpg"%(self.scene_id))
        
        #get scene PLY
        filename = self.root_dir + "/PointCloud%.1d.ply"%(self.scene_id)
        self.scene = o3d.io.read_point_cloud(filename)
        self.scene_points = np.asarray(self.scene.points)
        #extrinsic transformation
        self.scene_points = self.transform3D(self.scene_points, self.camera_pose)
        self.scene.points = o3d.utility.Vector3dVector(self.scene_points)
        
        #get bbox coordinates
        self.bboxes = np.genfromtxt(self.root_dir + "/%.1d/mask_out.txt"%(self.scene_id), dtype = 'str')[:,1:].astype(float)
        
        self.project(self.scene_points, self.rgb, self.camera_pose)
        
    def orthoProj(self, vecin, K):
        vecout = self.K@vecin.transpose(1,0)
        vecout[0,:] /= vecout[2,:]
        vecout[1,:] /= vecout[2,:]
        return vecout

    def getRot3D(self, pose):
        alpha, beta, gamma, x, y ,z = pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]
        s = np.pi/180
        alpha = alpha*s
        beta = beta*s
        gamma = gamma*s
        Rot_x = np.array([np.cos(alpha), np.sin(alpha), 0, -np.sin(alpha), np.cos(alpha), 0, 0, 0, 1]).reshape(3,3)
        Rot_y = np.array([np.cos(beta), 0 , np.sin(beta), 0, 1, 0, -np.sin(beta), 0, np.cos(beta)]).reshape(3,3)
        Rot_z = np.array([1,0,0,0, np.cos(gamma), np.sin(gamma), 0, -np.sin(gamma),  np.cos(gamma)]).reshape(3,3)
        Rot = Rot_x@Rot_y@Rot_z
        H = np.eye(4)
        H[:3,:3] = Rot
        H[:3, 3] = np.array([x,y,z])
        return H

    def transform3D(self, vecin, pose):
        H = self.camera_pose #self.getRot3D(pose)
        vecin_4 = np.hstack((np.asarray(vecin), np.ones([vecin.shape[0],1], vecin.dtype)))
        vecout = H@vecin_4.transpose(1,0)
        return vecout[:3,:].transpose(1,0)
    
    def project(self, points, rgb, ext_mat):
        
        #normalize
        x = points[:,0]/points[:,2]
        y = points[:,1]/points[:,2]
        
        #find radial distance
        r2 = x**2 + y**2
        
        #find tangential distortion
        xTD = 2*self.TangentialDistortion1*x*y + self.TangentialDistortion2*(r2 + 2*x**2)
        yTD = self.TangentialDistortion1*(r2 + 2*y**2) + 2*self.TangentialDistortion2*x*y
        
        #find radial distortion
        xRD = x*(1+ self.RadialDistortion1*r2 + self.RadialDistortion2*(r2**2) + self.RadialDistortion3*(r2**3))
        yRD = y*(1+ self.RadialDistortion1*r2 + self.RadialDistortion2*(r2**2) + self.RadialDistortion3*(r2**3))
        
        #undistort
        x_u = xRD + xTD
        y_u = yRD + yTD
        
        #project undistorted points to 3D
        points_undistorted = np.array([x_u*points[:,2], y_u*points[:,2], points[:,2]])
        
        #project using camera matrix
        projection = self.orthoProj(points_undistorted.transpose(1,0), self.K)
        
        #remove points outside image
        projection_x = np.where(projection[0,:]>self.width-1, self.width-1, projection[0,:]).astype(int)
        projection_y = np.where(projection[1,:]>self.height-1, self.height-1, projection[1,:]).astype(int)
        projection_x = np.where(projection_x<0, 0, projection_x)
        projection_y = np.where(projection_y<0, 0, projection_y)
        
        #set values
        self.Xpos[projection_y -1, self.width -projection_x - 1] = points_undistorted[0,:]
        self.Ypos[projection_y -1, self.width -projection_x - 1] = points_undistorted[1,:]
        self.Zpos[projection_y -1, self.width -projection_x - 1] = points_undistorted[2,:] 
            
    def clip_bolt_pcd(self, bbox_id):
        
        #get bbox position and masked pixel
        bbox = self.bboxes[bbox_id,:].astype(int) #top left x, y size x size y
        bbox[0] -= 144
        
        #get mask
        mask = (plt.imread(self.root_dir + "/%.1d/%.1d_mask.bmp"%(self.scene_id, bbox_id))>20)[:,:,0]
        mask_full = np.zeros([self.height, self.width])
        
        mask_full[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = mask[:bbox[3], :bbox[2]]
        #plt.imshow(mask_full*self.rgb[:,144:,0])
        
        #get masked cloud points
        bolt_indices = np.nonzero(self.Zpos*mask_full)
        bolt_points = np.array([self.Xpos[bolt_indices], self.Ypos[bolt_indices], self.Zpos[bolt_indices]]).transpose(1, 0)
        
        bolt_pcd = o3d.geometry.PointCloud()
        bolt_pcd.points = o3d.utility.Vector3dVector(bolt_points)
        
        bolt_pcd.paint_uniform_color([1,0,0])
        o3d.visualization.draw_geometries([self.scene, bolt_pcd])
        
        #bolt_pcd, _ = bolt_pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=0.5)
        
        return bolt_pcd
    


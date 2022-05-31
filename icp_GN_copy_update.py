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


class icp_GN():
    def __init__(self, grid_size, distance_threshold, iterations, mode, animate, model, scene):
        
        self.dist = distance_threshold
        self.mode = mode
        self.animate = animate
        
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
        self.normals = np.asarray(self.model.normals)
        self.model = self.model.voxel_down_sample(grid_size)
        self.tree = o3d.geometry.KDTreeFlann(self.model)
        
        # get scene downsampled
        self.scene = self.scene.voxel_down_sample(grid_size)
        
        self.reset()
        
        self.alignedCloud_left = o3d.geometry.PointCloud()
        self.alignedCloud_right = o3d.geometry.PointCloud()
        
        self.transformedCloud = o3d.geometry.PointCloud()
        

        #set static and dynamic starting clouds
        self.staticPointCloud = np.asarray(self.model.points).T
        self.dynamicPointCloud = np.asarray(self.scene.points).T
        
        #starting center points
        #self.staticMean = self.staticPointCloud.mean(axis=1).reshape(3,1)
        #self.dynamicMean = self.dynamicPointCloud.mean(axis=1).reshape(3,1)
        
        #scale points
        self.staticPointCloud *= self.scale 
        self.dynamicPointCloud *= self.scale
        
        #initial error
        self.error = np.linalg.norm(self.dynamicMean - self.staticMean)
        self.errors = []
        self.icp_transforms = []
        
        #parameters
        self.maxIterations = iterations
        self.numRandomSamples = 1000 #min(self.numDynamicPoints, self.numStaticPoints)
        self.eps = 1e-3
        self.cost = [1e09]
        
        #get sizes
        self.numDynamicPoints = self.dynamicPointCloud.shape[1]
        self.numStaticPoints = self.staticPointCloud.shape[1]
        
        self.weights = np.ones(self.numDynamicPoints)
        
        
    def display(self):
        # Create figure display
        self.fig = plt.figure()
        
        self.ax = plt.axes(xlim = [self.min[0],self.max[0]], projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_title("point-to-{}-GN".format(self.mode))
        utils.axisEqual3D(self.ax)
        self.static = self.ax.scatter(self.staticPointCloud[0,:], self.staticPointCloud[1,:], self.staticPointCloud[2,:], marker='.',color = "blue")
        self.ax.scatter(self.dynamicPointCloud[0,:], self.dynamicPointCloud[1,:], self.dynamicPointCloud[2,:], marker='.', color = "red")
        self.dynamic = self.ax.scatter(self.dynamicPointCloud[0,:], self.dynamicPointCloud[1,:], self.dynamicPointCloud[2,:], marker='.', color = "green")
        self.label = self.ax.text(-0.5, -0.5, 50, "Step = 0  Error: %.2d"%(self.error))
        self.lines = []
        self.projs = []
        
        '''
        for i in range(self.numRandomSamples):
            line, = self.ax.plot3D([self.staticPointCloud[0,i], self.dynamicPointCloud[0,i]],[self.staticPointCloud[1,i], self.dynamicPointCloud[1,i]],[self.staticPointCloud[2,i], self.dynamicPointCloud[2,i]], color = 'c')
            self.lines.append(line)
            
        for i in range(self.numRandomSamples):
            proj, = self.ax.plot3D([self.staticPointCloud[0,i], self.dynamicPointCloud[0,i]],[self.staticPointCloud[1,i], self.dynamicPointCloud[1,i]],[self.staticPointCloud[2,i], self.dynamicPointCloud[2,i]], color = 'k')
            self.projs.append(proj)
        '''
        
    def reset(self):
        #outputs
        #start of each iteration
        self.euler = np.zeros(6)
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self.euler_final = np.zeros(6)
        self.transformation = np.eye(4)
        
    def run_icp(self):
        self.reset()
        
        if self.mode == "point":
            if not self.animate:
                iter = 0
                while iter < self.maxIterations:
                    self.update_point_to_point(iter)
                    iter +=1
            else:
                func = FuncAnimation(self.fig, self.update_point_to_point, frames = self.maxIterations, interval = 1, repeat = False)
                plt.show()
                
        if self.mode == "plane":
            if not self.animate:
                iter = 0
                while iter < self.maxIterations:
                    self.update_point_to_plane(iter)
                    iter +=1
            else:
                func = FuncAnimation(self.fig, self.update_point_to_plane, frames = self.maxIterations, interval = 1, repeat = False)
                plt.show()
                
        print("final", self.euler_final, self.error)
        
        rot, trans = self.eulerToRot(self.euler_final)
        h = np.eye(4)
        h[:3,:3] = rot
        h[:3, 3] = trans.reshape(3)
        
        self.errors.append(self.error)
        self.icp_transforms.append(h)
        
        return h
        
                
    def alignment_left(self, R):
        
        #initialize start using eigenvalue decomp
        points = R@(np.asarray(self.scene.points)).transpose(1,0) 
        self.dynamic._offsets3d = (points[0,:], points[1,:], points[2,:])

        self.alignedCloud_left.points = o3d.utility.Vector3dVector(np.asarray(points.T))
        
    def alignment_right(self, R):
        
        #initialize start using eigenvalue decomp
        points = R@(np.asarray(self.scene.points)).transpose(1,0) 
        self.dynamic._offsets3d = (points[0,:], points[1,:], points[2,:])

        self.alignedCloud_right.points = o3d.utility.Vector3dVector(np.asarray(points.T))
        
                
    def shift_by_mean(self, cloud):
        points = np.asarray(cloud.points)
        mean = np.mean(points, axis=0)
        cloud_out = o3d.geometry.PointCloud()
        cloud_out.points = o3d.utility.Vector3dVector(points - mean)
        return cloud_out, mean
  
    def get_correspondences(self):
        #sample
        rand = random.choices(range(self.numDynamicPoints), k = self.numRandomSamples)
        p = self.dynamicPointCloud[:,rand]
        x = np.zeros_like(p)
        n = np.zeros_like(p)
        
        for i in range(self.numRandomSamples):
            _ , idx, dsq = self.tree.search_knn_vector_3d(p[:,i], 1)
            x[:,i] = self.staticPointCloud[:, idx[0]]
            n[:,i] = self.normals[idx[0], :]
            
        return p, x, n
    
    def update_point_to_point(self, framenumber):
        
        P, X, _ = self.get_correspondences()
        #######################################################################
        ###########################Gauss Newton################################
        
        #Jacobian
        H = np.zeros((6,6))
        b = np.zeros((6,1))
        Rot, trans = self.eulerToRot(self.euler)
        chi = 0
        
        for i in range(self.numRandomSamples):
            p,x = P[:,i], X[:,i]
            Rot_p = Rot@p
            e = (Rot_p + trans).reshape(3) - x
            
            J = self.Jacobian_point(Rot_p)
            H += J.T@J 
            b += J.T@(e.reshape(3,1))
            chi += np.linalg.norm(e)
            
            #print(i, e)
            #f = p[:,i] - e.reshape(3)
            #self.lines[i].set_data_3d([x[0,i], p[0,i]],[x[1,i], p[1,i]],[x[2,i], p[2,i]])
            #self.projs[i].set_data_3d([f[0], p[0,i]],[f[1], p[1,i]],[f[2], p[2,i]])
          
        update = - np.linalg.inv(H)@b
        self.euler = self.euler + update.reshape(6)
            
        #######################################################################
        #######################################################################        
        
        #update cloud for next iteration
        rotation, translation = self.eulerToRot(self.euler)
        self.dynamicPointCloud = rotation@(self.dynamicPointCloud) + translation.reshape(3,1)
        
        self.rotation = rotation@self.rotation
        self.translation = rotation@self.translation + translation.reshape(3,1)
        self.euler_final = self.rotToEuler(self.rotation, self.translation)
        
        #update plots
        self.label.set_text("Step = %.2d  Error = %.2f"%(framenumber, chi/self.numRandomSamples))
        self.dynamic._offsets3d = (P[0,:], P[1,:], P[2,:])
        #self.static._offsets3d = (x[0,:], x[1,:], x[2,:])
        
        self.error = chi/self.numRandomSamples
        
    def update_point_to_plane(self, framenumber):
        
        P, X, N = self.get_correspondences()
        #o3d.visualization.draw_geometries([self.model, self.scene])
        #######################################################################
        ###########################Gauss Newton################################
        
        #Jacobian
        H = np.zeros((6,6))
        b = np.zeros((6,1))
        Rot, trans = self.eulerToRot(self.euler)
        chi = 0
        
        for i in range(self.numRandomSamples):
            p,x,n = P[:,i], X[:,i], N[:,i]
            Rot_p = Rot@p
            e = (Rot_p.reshape(3) + trans - x).dot(n)
            J = self.Jacobian_plane(Rot_p, n)
            H += J.T@J 
            b += J.T*e
            chi += np.linalg.norm(e)
            
            #f = p[:,i] + e*n[:,i].reshape(3) #2*normal.reshape(3) #
            #print(mag, e, f.shape)
            #self.lines[i].set_data_3d([x[0,i], p[0,i]],[x[1,i], p[1,i]],[x[2,i], p[2,i]])
            #self.projs[i].set_data_3d([f[0], p[0,i]],[f[1], p[1,i]],[f[2], p[2,i]])
            
        update = - np.linalg.inv(H)@b
        self.euler = self.euler + update.reshape(6)
        
           
        #######################################################################
        #######################################################################        
        
        #update cloud for next iteration
        rotation, translation = self.eulerToRot(self.euler)
        
        self.dynamicPointCloud = rotation@(self.dynamicPointCloud) + translation.reshape(3,1)
        
        self.rotation = rotation@self.rotation
        self.translation = rotation@self.translation + translation.reshape(3,1)
        self.euler_final = self.rotToEuler(self.rotation, self.translation)
        
        #update plots
        self.label.set_text("Step = %.2d  Error = %.2f"%(framenumber, chi/self.numRandomSamples))
        self.dynamic._offsets3d = (P[0,:], P[1,:], P[2,:])
        #self.static._offsets3d = (x[0,:], x[1,:], x[2,:])
        
        self.error = chi/self.numRandomSamples
            
    
        
    def Jacobian_point(self, Rot_p):
        J = np.zeros((3,6))
        J[:, 3:] = np.eye(3)
        J[:,0] = np.cross(Rot_p, np.array([1, 0, 0]))
        J[:,1] = np.cross(Rot_p, np.array([0, 1, 0]))
        J[:,2] = np.cross(Rot_p, np.array([0, 0, 1]))
        return J
    
    def Jacobian_plane(self, Rot_p, n):
        J = np.zeros((1,6))
        J[:, 3:] = n
        J[:,:3] = -np.cross(Rot_p, n) 
        return J
    
    def rotToEuler(self, R, t):
        x,y,z = t[0,0], t[1,0], t[2,0]
        cs = np.linalg.norm(R[:2,0])
        if cs < 1e-16:
            alpha = np.arctan2(-R[1,2], R[1,1])
            beta = np.arctan2(-R[1,2], cs)
            gamma = 0
        else:
            alpha = np.arctan2(R[2,1], R[2,2])
            beta = np.arctan2(-R[2,0], cs)
            gamma = np.arctan2(R[1,0], R[0,0])
        
        
        return np.array([alpha, beta, gamma, x, y, z])
  
    def eulerToRot(self, euler):
        alpha, beta, gamma, x, y, z = euler[0], euler[1], euler[2], euler[3], euler[4], euler[5]

        Rot_z = np.array([np.cos(gamma), np.sin(gamma), 0, -np.sin(gamma), np.cos(gamma), 0, 0, 0, 1]).reshape(3, 3)
        Rot_y = np.array([np.cos(beta), 0, -np.sin(beta), 0, 1, 0, np.sin(beta), 0, np.cos(beta)]).reshape(3, 3)
        Rot_x = np.array([1, 0, 0, 0, np.cos(alpha), np.sin(alpha), 0, -np.sin(alpha),  np.cos(alpha)]).reshape(3, 3)
        Rot = Rot_z@Rot_y@Rot_x
        return Rot, np.array([x,y,z])

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
        
        #plot the axes for model and scene
        end = 50*v1
        self.ax.plot3D([0, end[0]], [0, end[1]], [0, end[2]], color = 'b' )
        
        end = 50*v2
        self.ax.plot3D([0, end[0]], [0, end[1]], [0, end[2]], color = 'r' )
        
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
        
        #plot rotated cloud axis
        end = 50*(R@v2).reshape(3)
        self.ax.plot3D([0, end[0]], [0, end[1]], [0, end[2]], color = 'g' )
        
        #plot rotation axis
        end = 50*ax.reshape(3)
        self.ax.plot3D([0, end[0]], [0, end[1]], [0, end[2]], color = 'k' )
    
        return R, R_eta
    
    def get_final_transformation(self):
        
        H1 = np.eye(4)
        H1[:3,3] = self.dynamicMean
        H4 = np.eye(4)
        H4[:3,3] = -self.staticMean

        H2_l = H2_r = np.eye(4)
        self.display()
        R1, R2= self.eigenvalue_init()
        H2_l[:3,:3] = R1
        H2_r[:3,:3] = R2
        
        #left
        
        self.alignment_left(R1)
        self.dynamicPointCloud = np.asarray(self.alignedCloud_left.points).T
        H3_l = self.run_icp()
           
        
        #right
        self.display()
        self.alignment_right(R2)
        self.dynamicPointCloud = np.asarray(self.alignedCloud_right.points).T
        H3_r = self.run_icp()
        
        #final transformation
        if self.errors[0] < self.errors[1]:
            H = H4@H3_l@H2_l@H1
        else:
            H = H4@H3_r@H2_r@H1
        
        return H

import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from scipy.optimize import minimize
import warnings
from sklearn.mixture import GaussianMixture

from my_utils.utils_calc import *


# ------------------------------------
class CableSensing(object):
    def __init__(self, b_visualize=False, open3d_viewpoint=None):
        self.task_id = 2
        self.open3d_viewpoint = open3d_viewpoint

        self.gelsight_img_height = 240
        self.gelsight_img_width = 320
        self.gel_height = 14.3 / 1000.0  # unit: m
        self.gel_width = 18.6 / 1000.0  # unit: m
        self.depth_value_rescale_factor = 0.1 / 1000.0 # unit: m (approximate)

        self.visualize = b_visualize
        self.b_vis_gmm = False
        self.b_vis_contact_mask = False
        self.init_vis = False

        self.gmm_depth_0 = GaussianMixture(n_components=2, tol=1e-3, max_iter=100, warm_start=False, init_params="k-means++", random_state=1)
        self.gmm_depth_1 = GaussianMixture(n_components=2, tol=1e-3, max_iter=100, warm_start=False, init_params="k-means++", random_state=1)

        self.depth_gmm_resize_factor = 10
        self.connected_region_min_area = 300
        self.contact_points_resize_factor = 4
        self.plane_min_diff = 1.0
        self.min_first_contact_points = 50 / (self.contact_points_resize_factor**2)
        self.min_contact_points = 1000 / (self.contact_points_resize_factor**2)
        self.pca_eigen_factor = 7
        self.min_z_dist_between_two_tactile = 0.004
        self.max_z_dist_between_two_tactile = 0.006
        self.bottom_mask_region = 64

        self.b_opt_use_jaco = True
        self.b_opt_warm_start = True
        self.last_theta = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # a, b, c, x0, y0, z0, radius, trans_x, trans_y, trans_z

        self.time_cost_opt = 0.0


    # --------------------------------
    def initVisualizer(self, points=None):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)

        if points is None:
            x = self.gel_height / 2.0
            y = self.gel_width / 2.0
            z = 1.0 / 1000.0
            points = np.array([[-x, -y, -z],
                               [-x, y, -z],
                               [-x, -y, z],
                               [-x, y, z],
                               [x, -y, -z],
                               [x, y, -z],
                               [x, -y, z],
                               [x, y, z]])
            
        # Create Open3D PointCloud for the pointcloud of contact region
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.vis.add_geometry(self.pcd)

        # Create Open3D LineSet for the axis
        axis_line_set = o3d.geometry.LineSet()
        axis_points = 0.005 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        axis_line_set.points = o3d.utility.Vector3dVector(axis_points)
        axis_lines = [[0, 1], [0, 2], [0, 3]]
        axis_line_set.lines = o3d.utility.Vector2iVector(axis_lines)
        axis_line_set.colors = o3d.utility.Vector3dVector([[1,0,0], [0,1,0], [0,0,1]])
        self.vis.add_geometry(axis_line_set)

        # Create Open3D LineSet for the second axis
        self.axis_line_set_2 = o3d.geometry.LineSet()
        self.axis_points_2 = 0.005 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.axis_line_set_2.points = o3d.utility.Vector3dVector(self.axis_points_2)
        axis_lines_2 = [[0, 1], [0, 2], [0, 3]]
        self.axis_line_set_2.lines = o3d.utility.Vector2iVector(axis_lines_2)
        self.axis_line_set_2.colors = o3d.utility.Vector3dVector([[1,0,0], [0,1,0], [0,0,1]])
        self.vis.add_geometry(self.axis_line_set_2)

        # Create Open3D LineSet for the DLO state
        self.dlo_line_set = o3d.geometry.LineSet()
        dlo_points = np.array([[0, 0, 0], [0, 0, 0]])
        self.dlo_line_set.points = o3d.utility.Vector3dVector(dlo_points)
        dlo_lines = [[0, 1]]
        self.dlo_line_set.lines = o3d.utility.Vector2iVector(dlo_lines)
        self.dlo_line_set.colors = o3d.utility.Vector3dVector([[1,0,0], [0,1,0], [0,0,1]])
        self.vis.add_geometry(self.dlo_line_set)

        view_control = self.vis.get_view_control()

        if self.open3d_viewpoint == 0:
            view_control.rotate(900.0, 0.0) # cable sensing 0
        else:
            view_control.rotate(520.0, 200.0) # cable sensing 1

    
    # --------------------------------
    def getDepthToContactMask(self, depth_img, gelsight_id):
        """
            Get the contact mask based on the depth_img
        """
        contact_mask = np.zeros(depth_img.shape, dtype=np.uint8)

        depth_data = depth_img.copy()
        data = depth_data[::self.depth_gmm_resize_factor, ::self.depth_gmm_resize_factor].reshape(-1, 1)

        if data.size < 10:
            return contact_mask, 0

        if gelsight_id == 0:
            gmm = self.gmm_depth_0
        elif gelsight_id == 1:
            gmm = self.gmm_depth_1

        # using GMM, get the contact mask by estimated depth threshold
        gmm.fit(data)
        mean1, mean2 = gmm.means_
        weight1, weight2 = gmm.weights_
        cov1, cov2 = gmm.covariances_
        if weight1 / np.sqrt(2 * np.pi * cov1) < weight2 / np.sqrt(2 * np.pi * cov2): # 这里使用weight的大小作为判断依据
            mean1, mean2 = mean2, mean1
            weight1, weight2 = weight2, weight1
            cov1, cov2 = cov2, cov1
        plane_depth = mean1[0]
        depth_thres = plane_depth + max(self.plane_min_diff, 3 * np.sqrt(cov1))

        contact_mask[depth_img > depth_thres] = 255

        if self.b_vis_gmm:
            x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
            pdf = np.exp(gmm.score_samples(x))
            pdf1 = weight1 * np.exp(-(x - mean1)**2 / (2 * cov1)) / np.sqrt(2 * np.pi * cov1)
            pdf2 = weight2 * np.exp(-(x - mean2)**2 / (2 * cov2)) / np.sqrt(2 * np.pi * cov2)

            plt.cla()
            plt.hist(data, bins=30, density=True, color='blue', alpha=0.7, label='Histogram')

            plt.plot(x, pdf1, color='red', linestyle='dashed', linewidth=2, label='Gaussian 1')
            plt.plot(x, pdf2, color='green', linestyle='dashed', linewidth=2, label='Gaussian 2')

            plt.title('Distribution Histogram with GMM Components')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)

            plt.draw()
            plt.pause(0.001)
            
        # remove the connected region with too small area
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(contact_mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, -1]
            centroid = centroids[i] # (x, y)
            if area < self.connected_region_min_area:
                contact_mask[labels==i] = 0

        # close operation
        kernel = np.ones((10, 10), np.uint8)
        contact_mask = cv2.morphologyEx(contact_mask, cv2.MORPH_CLOSE, kernel)

        return contact_mask, plane_depth
    

    # --------------------------------
    def transformPointsFromImageFrameToUserFrame(self, points):
        transformed_points = np.zeros(points.shape)
        transformed_points[:, 0] = (1.0/2.0 - points[:, 1] / self.gelsight_img_height) * self.gel_height
        transformed_points[:, 1] = (-1.0/2.0 + points[:, 0] / self.gelsight_img_width) * self.gel_width

        return transformed_points


    # --------------------------------
    def depthToPointcloud(self, depth_map):
        img_height, img_width = depth_map.shape[0], depth_map.shape[1]  
        x = np.arange(img_height)
        y = np.arange(img_width)
        X, Y = np.meshgrid(x, y) 

        points = np.zeros([img_height * img_width, 3])
        points[:, 0] = (1.0/2.0 - X.reshape(-1, ) / img_height) * self.gel_height
        points[:, 1] = (-1.0/2.0 + Y.reshape(-1, ) / img_width) * self.gel_width
        points[:, 2] = -(depth_map.T).reshape(-1, ) * self.depth_value_rescale_factor

        return points


    # --------------------------------
    def depthToContactPoints(self, depth_map, contact_mask):
        """
            Get the contact points in Gelsight frame given the depth_map and contact_mask
        """
        depth_map = depth_map[::self.contact_points_resize_factor, ::self.contact_points_resize_factor]
        contact_mask = contact_mask[::self.contact_points_resize_factor, 
                                    ::self.contact_points_resize_factor].T.reshape(-1, )

        all_points = self.depthToPointcloud(depth_map)
        contact_points = all_points[contact_mask == 255]

        return contact_points
    

    # --------------------------------
    def imageDownSampling(self, image, factor):
        """
            Input:
                factor: must be integar
        """
        return image[::factor, ::factor]
    

    # --------------------------------
    def optimization(self, contact_points_0, contact_points_1=None, radius=None):
        rescale_factor = 1000.0  # rescale the unit of points from m to mm
        weight_radius = 0.0001
        weight_axis = 0.0001
        weight_trans = 0.0005
        trans_x_min = -0.002 * rescale_factor
        trans_x_max = 0.002 * rescale_factor
        trans_y_min = -0.005 * rescale_factor
        trans_y_max = 0.005 * rescale_factor
        trans_z_min = 0 * rescale_factor
        trans_z_max = 0.002 * rescale_factor
        
        if contact_points_1 is not None:
            contact_points_0 = contact_points_0.copy() * rescale_factor
            contact_points_1 = contact_points_1.copy() * rescale_factor
            contact_points = np.vstack([contact_points_0, contact_points_1])
            z0_min = np.min(contact_points[:, 2])
            z0_max = np.max(contact_points[:, 2])
        else:       
            contact_points_0 = contact_points_0.copy() * rescale_factor
            contact_points = contact_points_0
            z0_min = np.min(contact_points_0[:, 2])
            z0_max = 0.01 * rescale_factor
        radius_min = 0.0
        radius_max = 0.01 * rescale_factor
        if radius is not None:
            radius = radius * rescale_factor 
            radius_max = radius * 1.2

        n_points = contact_points.shape[0]
        n_points_0 = contact_points_0.shape[0]
        n_points_1 = contact_points_1.shape[0] if contact_points_1 is not None else 0
        mean_points = np.mean(contact_points, axis=0)

        y0_min = np.min(contact_points[:, 1])
        y0_max = np.max(contact_points[:, 1])

        t_factor = np.array([0]*n_points_0 + [1]*n_points_1).reshape(-1, 1, 1)

        def skew(a):
            a = a.reshape(-1, 3)
            A = np.zeros((a.shape[0], 3, 3))
            A[:, 0, 1] = -a[:, 2]
            A[:, 0, 2] = a[:, 1]
            A[:, 1, 0] = a[:, 2]
            A[:, 1, 2] = -a[:, 0]
            A[:, 2, 0] = -a[:, 1]
            A[:, 2, 1] = a[:, 0]
            return A

        def objectFunction(theta):
            d_norm = np.linalg.norm(theta[0:3])
            d = np.repeat(theta[0:3].reshape(1, -1), n_points, axis=0)
            A = np.repeat(theta[3:6].reshape(1, -1), n_points, axis=0)
            t = np.concatenate([np.repeat(np.zeros((1, 3)), n_points_0, axis=0), 
                                np.repeat(theta[7:10].reshape(1, -1), n_points_1, axis=0)], axis=0)
            P = contact_points + t
            v = P - A
            D = np.linalg.norm(np.cross(v, d), axis=1, keepdims=True) / d_norm
            e = (D - theta[6]) if radius is None else (D - radius)

            cost_err = 1.0/2.0 * 1.0/n_points * e.T @ e
            cost_radius = 1.0/2.0 * weight_radius * theta[6]**2
            axis_y_z = np.array([0, theta[1], theta[2]]).reshape(-1, 1)
            cost_axis = 1.0/2.0 * weight_axis * axis_y_z.T @ axis_y_z
            trans = theta[7:10].reshape(-1, 1)
            cost_trans = 1.0/2.0 * weight_trans * trans.T @ trans

            cost = cost_err + cost_radius + cost_axis + cost_trans
            return cost[0, 0]
        
        def jacobian(theta):
            d_norm = np.linalg.norm(theta[0:3])
            d = np.repeat(theta[0:3].reshape(1, -1), n_points, axis=0) # n*3
            d_T = np.expand_dims(d, axis=1) # n*1*3
            A = np.repeat(theta[3:6].reshape(1, -1), n_points, axis=0)
            t = np.concatenate([np.repeat(np.zeros((1, 3)), n_points_0, axis=0), 
                                np.repeat(theta[7:10].reshape(1, -1), n_points_1, axis=0)], axis=0)
            P = contact_points + t
            v = P - A
            u = np.cross(v, d) # n*3
            u_norm = np.linalg.norm(u, axis=1, keepdims=True) # n*1
            u_normalized = u / u_norm # n*3
            u_normalized_T = np.expand_dims(u_normalized, axis=1) # n*1*3
            D = u_norm / d_norm
            e = (D - theta[6]) if radius is None else (D - radius)

            axis_y_z = np.array([0, theta[1], theta[2]]).reshape(-1, 1)
            trans = theta[7:10].reshape(-1, 1)

            dD_dd = (np.einsum('ijk,ikl->ijl', u_normalized_T, skew(v)) * d_norm - u_norm.reshape(-1,1,1) * d_T/d_norm ) / (d_norm**2) # n*1*3
            dD_dA = np.einsum('ijk,ikl->ijl', u_normalized_T, skew(d)) / (d_norm) # n*1*3
            dD_dt = -np.einsum('ijk,ikl->ijl', u_normalized_T, skew(d)) / (d_norm) * t_factor # n*1*3
            dC_dd = 1.0/n_points * e.T @ dD_dd.reshape(-1, 3) + weight_axis * axis_y_z.T # 1*3
            dC_dA = 1.0/n_points * e.T @ dD_dA.reshape(-1, 3) # 1*3
            dC_dr = -1.0/n_points * e.T @ np.ones((n_points, 1))  \
                        + weight_radius * theta[6] # 1*1
            dC_dt = 1.0/n_points * e.T @ dD_dt.reshape(-1, 3) + weight_trans * trans.T  # 1*3
            
            jaco_list = [dC_dd, dC_dA, dC_dr, dC_dt]
            return np.hstack(jaco_list).reshape(-1, )
        
        def thetaEqConstraint(theta):
            d = theta[0:3]
            return np.dot(d, d) - 1 # = 0
        
        def x0EqConstraint(theta):
            # x0 = mean_x
            # trans_x = 0
            eq = theta[3] - np.array([mean_points[0]])
            return eq.reshape(-1, )
        
        constraints_list = [dict(type='eq', fun=thetaEqConstraint),
                            dict(type='eq', fun=x0EqConstraint)]

        bounds_list = [(0.5, 1), (-1, 1), (-0.2, 0.2), (None, None), (y0_min, y0_max), (z0_min, z0_max), (radius_min, radius_max), \
                       (trans_x_min, trans_x_max), (trans_y_min, trans_y_max), (trans_z_min, trans_z_max)]
        
        theta_init = self.last_theta
        
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        res = minimize(fun=objectFunction, 
                       jac=jacobian if self.b_opt_use_jaco else None,
                       x0=theta_init, 
                       constraints=constraints_list,
                       bounds=bounds_list,
                       method='SLSQP', options={'ftol':1e-10}
                       )  
        res_theta = res.x.reshape(-1, )
        if self.b_opt_warm_start:
            self.last_theta = res_theta.copy()

        res_theta[3:10] /= rescale_factor
        return res_theta


    # --------------------------------
    def getCableInhandState(self, depth_map_0, 
                            depth_map_1=None, tactile_1_pose_in_tactile_0=None, 
                            marker_pos_0=None, marker_pos_init_0=None,
                            marker_pos_1=None, marker_pos_init_1=None):
        """
            Input:
                tactile_1_pose_in_tactile_0: 4*4 matrix
        """
        
        contact_mask_0, plane_depth = self.getDepthToContactMask(depth_map_0, gelsight_id=0)
        contact_points_0 = self.depthToContactPoints(depth_map_0, contact_mask_0)
        contact_points = contact_points_0
        normal_force_0 = self.getContactNormalForce(depth_map_0, contact_mask_0, plane_depth)
        if marker_pos_0 is not None and marker_pos_init_0 is not None:
            shear_force_0, shear_moment_0 = self.getContactShearForce(marker_pos_0, marker_pos_init_0, contact_mask_0)

        if depth_map_1 is not None:
            # clip the z-axis position of tactile_1_pose_in_tactile_0 into a pre-defined range (due to the imprecise hand FK)               
            tactile_1_pose_in_tactile_0[2, 3] = np.clip(tactile_1_pose_in_tactile_0[2, 3], 
                                                        self.min_z_dist_between_two_tactile, 
                                                        self.max_z_dist_between_two_tactile)

            contact_mask_1, plane_depth = self.getDepthToContactMask(depth_map_1, gelsight_id=1)
            contact_points_1 = self.depthToContactPoints(depth_map_1, contact_mask_1)
            contact_points_1 = transformPositions(positions=contact_points_1, 
                                                  target_frame_pose_inv=tactile_1_pose_in_tactile_0)
            contact_points = np.vstack([contact_points_0, contact_points_1])
            normal_force_1 = self.getContactNormalForce(depth_map_1, contact_mask_1, plane_depth)
            if marker_pos_1 is not None and marker_pos_init_1 is not None:
                shear_force_1, shear_moment_1 = self.getContactShearForce(marker_pos_1, marker_pos_init_1, contact_mask_1)
        else:
            contact_points_1 = None

        if self.b_vis_contact_mask:
            cv2.imshow('depth_map_0', depth_map_0 /10.0)
            cv2.imshow('contact_mask_0', contact_mask_0)
            if depth_map_1 is not None:
                cv2.imshow('depth_map_1', depth_map_1 /10.0)
                cv2.imshow('contact_mask_1', contact_mask_1)
            cv2.waitKey(1)

        if depth_map_1 is not None:
            b_in_contact = (self.goodDLOContact(contact_points_0) or self.goodDLOContact(contact_points_1) or self.goodDLOContact(contact_points)) \
                and contact_points.shape[0] > self.min_contact_points
        else:
            b_in_contact = self.goodDLOContact(contact_points) and contact_points.shape[0] > self.min_contact_points

        # estimate the main axis
        if b_in_contact:
            t1 = time.time()
            line_theta = self.optimization(contact_points_0, contact_points_1, radius=None)
            self.time_cost_opt += time.time() - t1
            main_axis = line_theta[0:3] # [a, b, c]
            via_point = line_theta[3:6] # [x0, y0, z0]
            radius = line_theta[6]
            trans = line_theta[7:10]

            contact_points = np.vstack([contact_points_0, contact_points_1 + trans])
            tactile_1_pose_in_tactile_0 = tactile_1_pose_in_tactile_0.copy()
            tactile_1_pose_in_tactile_0[0:3, 3] += trans

        # visualization by open3d
        if self.visualize:
            if self.init_vis == False:
                self.initVisualizer()
                self.init_vis = True

            self.pcd.points = o3d.utility.Vector3dVector(contact_points)
            self.vis.update_geometry(self.pcd)

            if b_in_contact:
                dlo_point_0 = via_point - 0.1 * main_axis
                dlo_point_1 = via_point + 0.1 * main_axis
                dlo_points = np.concatenate([dlo_point_0.reshape(1, -1), 
                                            dlo_point_1.reshape(1, -1)], axis=0)
            else:
                dlo_points = np.array([[0, 0, 0], [0, 0, 0]])
            self.dlo_line_set.points = o3d.utility.Vector3dVector(dlo_points)
            self.vis.update_geometry(self.dlo_line_set)

            if depth_map_1 is not None:
                # update tactile_1 frame axis
                axis_points = transformPositions(self.axis_points_2, 
                                                 target_frame_pose_inv=tactile_1_pose_in_tactile_0)
                self.axis_line_set_2.points = o3d.utility.Vector3dVector(axis_points)
                self.vis.update_geometry(self.axis_line_set_2)
            
            self.vis.poll_events()
            self.vis.update_renderer()

        res_dict = {}
        res_dict["contact_points_0"] = contact_points_0
        res_dict["normal_force_0"] = normal_force_0
        if depth_map_1 is not None:
            res_dict["contact_points_1"] = contact_points_1
            res_dict["normal_force_1"] = normal_force_1
        if marker_pos_0 is not None:
            res_dict["shear_force_0"] = shear_force_0
            res_dict["shear_moment_0"] = shear_moment_0
        if marker_pos_1 is not None:
            res_dict["shear_force_1"] = shear_force_1
            res_dict["shear_moment_1"] = shear_moment_1

        res_dict["b_contact_0"] = contact_points_0.shape[0] > self.min_first_contact_points
        res_dict["b_contact_1"] = contact_points_1.shape[0] > self.min_first_contact_points
        res_dict["b_good_contact"] = b_in_contact
        res_dict["b_good_two_contact"] = b_in_contact and contact_points_0.shape[0] > self.min_contact_points and contact_points_1.shape[0] > self.min_contact_points

        if b_in_contact:
            res_dict["main_axis"] = main_axis
            res_dict["via_point"] = via_point
            res_dict["radius"] = radius

            px = 0.0
            py = via_point[1] + main_axis[1] / main_axis[0] * (px - via_point[0])
            pz = via_point[2] + main_axis[2] / main_axis[0] * (px - via_point[0])
            rx = 0
            rz = np.arctan2(main_axis[1], main_axis[0])
            ry = -np.arctan2(main_axis[2], np.sqrt(main_axis[0]**2  + main_axis[1]**2))
            inhand_pose = np.array([px, py, pz, rx, ry, rz])
            res_dict["inhand_dlo_pose"] = inhand_pose

        return res_dict
    

    # --------------------------------
    def getContactNormalForce(self, depth_map, contact_mask, zero_depth):
        contact_depth = depth_map[contact_mask != 0]
        normal_force = (np.mean(contact_depth) - zero_depth) if contact_depth.size != 0 else 0
        return normal_force
    

    # --------------------------------
    def getContactShearForce(self,  marker_pos, marker_pos_init, contact_mask):
        marker_pos = np.asarray(marker_pos).reshape(-1, 2)
        marker_pos_init = np.asarray(marker_pos_init).reshape(-1, 2)

        kernel = np.ones((10,10), np.uint8)
        contact_mask = cv2.dilate(contact_mask, kernel, iterations=1)

        contact_marker_indices = []
        for i in range(marker_pos.shape[0]):
            col = int(np.clip(marker_pos[i, 0], 0, self.gelsight_img_width-1))
            row = int(np.clip(marker_pos[i, 1], 0, self.gelsight_img_height-1))
            if contact_mask[row, col] != 0:
                contact_marker_indices.append(i)

        if len(contact_marker_indices) == 0:
            force = np.array([0, 0])
            moment = 0
        else:
            contact_marker_pos = self.transformPointsFromImageFrameToUserFrame(
                                                marker_pos[contact_marker_indices]) * 1000.0
            contact_marker_pos_init = self.transformPointsFromImageFrameToUserFrame(
                                                marker_pos_init[contact_marker_indices]) * 1000.0

            marker_pos_displacements = contact_marker_pos - contact_marker_pos_init
            force = np.mean(marker_pos_displacements, axis=0)
            moment = np.mean(np.cross(contact_marker_pos_init, marker_pos_displacements).reshape(-1, 1), axis=0)

        return force, moment


    # --------------------------------
    def goodContact(self, depth_map_0, depth_map_1):
        if np.count_nonzero(depth_map_0 > 2.0) > 100 \
            and np.count_nonzero(depth_map_1 > 2.0) > 100:
            return True
        else:
            return False


    # --------------------------------
    def goodDLOContact(self, contact_points):
        """
            Using PCA to determine whether the contact between DLO and tactile sensor is good.
        """
        if contact_points.shape[0] < self.min_contact_points:
            return False
        
        data = contact_points
        
        # Standardize the data (optional but often recommended)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # standardized_data = (data - mean) / std
        standardized_data = (data - mean)

        # Compute the covariance matrix
        covariance_matrix = np.cov(standardized_data, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # According to the largest eigenvalue and second largest eigenvalue
        return eigenvalues[0] > self.pca_eigen_factor * eigenvalues[1]

import numpy as np
import math
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.cluster import DBSCAN
#import open3d

# A label of -1 corresponds to noise
def cluster_point_cloud(pts, eps=0.3, min_samples=30):
	db = DBSCAN(eps=0.3, min_samples=10, algorithm='ball_tree', n_jobs=2).fit(pts)
	labels = db.labels_
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	unique_labels = set(labels)
	return n_clusters, labels

# Assumes n_clusters > 1
def filter_clusters(n_clusters, labels, bb_pts, box, f_y):
	h_guess = 2.0
	n_pts = np.zeros((1, n_clusters))
	d_pts = np.zeros((1, n_clusters))
	z_pts = np.zeros((1, n_clusters))
	for i in range(0,n_clusters):
		cluster_idx = [j for j in range(0, bb_pts.shape[0]) if labels[j] == i]
		cluster_pts = bb_pts[cluster_idx,:]
		n_pts[0,i] = len(cluster_idx)
		d_pts[0,i] = np.mean(cluster_pts[:,2])
		z_pts[0,i] = d_pts[0,i] * box[3] / f_y

	err = np.abs(z_pts - h_guess)
	idx = np.argmax(n_pts)
	cluster_idx = [j for j in range(0, bb_pts.shape[0]) if labels[j] == idx]
	if 1.5 <= z_pts[0,idx] <= 3:
		return idx, cluster_idx
	else:
		return np.argmin(err), cluster_idx

# velo: (n,4) numpy array
def create_bev(velo, h_metric, w_metric, meter_to_pixel):
	# Assume we want the middle, center of image as velodyne center
	w = w_metric * meter_to_pixel
	h = h_metric * meter_to_pixel
	bev = np.ones((h, w, 3), dtype=np.uint8) * 255		# depth: intensity
	u = (h - velo[:,2]*meter_to_pixel).astype(int)
	v = ((w / 2) + velo[:,0]*meter_to_pixel).astype(int)
	q = (u < h) * (u >= 0) * (v < w) * (v >= 0)
	indices = np.where(q)[0]
	u = u[indices]
	v = v[indices]
	bev[np.ravel(u),np.ravel(v),:] = 0
	return bev

def passthrough(velo, xmin, xmax, ymin, ymax, zmin, zmax):
	q = (xmin < velo[:,0]) * (velo[:,0] < xmax) * \
		(ymin < velo[:,1]) * (velo[:,1] < ymax) * \
		(zmin < velo[:,2]) * (velo[:,2] < zmax)
	indices = np.where(q)[0]
	return velo[indices,:]

def passthrough2(velo, xmin, xmax, ymin, ymax):
	q = (xmin < velo[:,0]) * (velo[:,0] < xmax) * \
		(ymin < velo[:,1]) * (velo[:,1] < ymax)
	indices = np.where(q)[0]
	return velo[indices,:], indices

#def visualizeCloud(velo):
#	pcd = open3d.PointCloud()
#	pcd.points = open3d.Vector3dVector(velo[:,0:3])
#	open3d.draw_geometries([pcd])

def rgba2rgb(rgba):
	alpha = rgba[3]
	r = alpha * rgba[0] * 255
	g = alpha * rgba[1] * 255
	b = alpha * rgba[2] * 255
	return (r,g,b)

def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']

    return points
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: a 4x2 matrix of coordinates
    :rval: (4,2)
    
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.
   
    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]
  
    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r) 
    return rval

def points2centroid_yaw(p1, p2, p3, p4):
    """
    Takes in four points of a rectangle continuous s.t. p1 ->p2->p3->p4 traces a rectangle
    Returns centroid and yaw measured from positive x axis from (-pi/2 to pi/2); tilted to right is negative yaw

    :param points: 4 points in (x,y); each point can be tuple, list or numpy array
    :rval: width, length, centroid_x, centroid_y , yaw in radians
    
    """ 
    cx = (p1[0]+p2[0]+p3[0]+p4[0])/4    
    cy = (p1[1]+p2[1]+p3[1]+p4[1])/4
    
    # to find yaw, first find the longer side of the box
    side1 = np.array([[p2[0] - p1[0]], [p2[1] - p1[1]]])
    side2 = np.array([[p3[0] - p2[0]], [p3[1] - p2[1]]])
    if(np.linalg.norm(side1) > np.linalg.norm(side2)):
      longside = side1
      w = np.linalg.norm(side2)
      l = np.linalg.norm(side1)
    else:
      longside = side2
      w = np.linalg.norm(side1)
      l = np.linalg.norm(side2)

    yaw = math.atan(longside[1]/longside[0])
    return w, l, cx, cy, yaw

def centroid_yaw2points(w, l, cx, cy, yaw):
    """
    Takes in centroid and yaw measured from positive x axis from (-pi/2 to pi/2); tilted to right is negative yaw
    Returns four points of a rectangle continuous s.t. p1 ->p2->p3->p4 traces a rectangle

    :param points: width, length, centroid_x, centroid_y , yaw in radians
    :rval: 4 points in (x,y) rerpesented as numpy arrays
    
    """ 
    box_at_origin = np.matrix([ [w/2, -l/2],
                               [w/2,  l/2],
                               [-w/2, l/2],
                               [-w/2, -l/2] ])


    inv_rotate = np.matrix([[np.cos(yaw), -np.sin(yaw)],     
                        [np.sin(yaw), np.cos(yaw)]])

    box_rotated = box_at_origin * inv_rotate
    box_final = np.squeeze(np.asarray(box_rotated + [cx, cy])) 
    
    return box_final[0], box_final[1], box_final[2], box_final[3] 


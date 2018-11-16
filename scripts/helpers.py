import numpy as np
import math
from scipy.spatial import ConvexHull

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


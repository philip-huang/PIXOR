from os.path import join
import cv2
import numpy as np
from helpers import *

meter_to_pixel = 20
BEV_H = 40
BEV_W = 30

def plot_image(bag, name):
    file_name = '/home/briallan/HawkEyeData/TestData/_2018-10-30-15-25-07/velo/'+name + '.png'
    #file_name = join(bag, 'velo', name+'.png')
    im = cv2.imread(file_name)
    cv2.imshow('original', im)
    cv2.waitKey(0)
    # rotate image
    #im = cv2.transpose(im)
    #im = cv2.flip(im, 1)

    #label_name = join(bag, 'labels', name+'.txt')
    label_name = '/home/briallan/HawkEyeData/TestData/_2018-10-30-15-25-07/labels/'+name + '.txt'
    
    with open(label_name, 'r') as f:
        lines = f.readlines() 
        label_list = []
        for line in lines:
            bbox = np.zeros((4, 2))
            entry = line.split(' ')
            object = entry[0]
            w = float(entry[1])
            l = float(entry[2])
            cx = float(entry[3]) 
            cy = float(entry[4]) 
            yaw = float(entry[5]) 
            print(object, w, l, cx, cy, yaw)
            
            bbox[0], bbox[1], bbox[2], bbox[3] = centroid_yaw2points(w, l, cx, cy, yaw)
            label_list.append(bbox)
    print("Labelled boxes:", label_list)
    for corners in label_list:  
        corners = corners* meter_to_pixel
        plot_corners = corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(im, [plot_corners], True, (255, 0, 0), 2)

    cv2.imshow('gt', im)
    cv2.waitKey(0)
    #plot_bev(im, label_list, map_height=BEV_H * meter_to_pixel)

        
            
            
def plot_bev(velo_array, label_list = None, map_height=600, window_name='GT'):
    '''
    Plot a Birds Eye View Lidar and Bounding boxes (Using OpenCV!)
    The heading of the vehicle is marked as a red line
        (which connects front right and front left corner)

    :param velo_array: a 2d velodyne points
    :param label_list: a list of numpy arrays of shape [4, 2], which corresponds to the 4 corners' (x, y)
    The corners should be in the following sequence:
    rear left, rear right, front right and front left
    :param map_height: height of the map
    :param window_name: name of the open_cv2 window
    :return: None
    '''
 

    if label_list is not None:
        for corners in label_list:
            plot_corners = corners * meter_to_pixel
            plot_corners[:, 1] += int(map_height//2)
            plot_corners[:, 1] = map_height - plot_corners[:, 1]
            print(plot_corners/meter_to_pixel)
            plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(velo_array, [plot_corners], True, (255, 0, 0), 2)
            cv2.line(velo_array, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 0, 255), 3)

    cv2.imshow(window_name, velo_array)
    cv2.waitKey(0)
        
if __name__ == "__main__":  
    bag = '_2018-10-30-15-25-07'
    name = 'bev0000'
    plot_image(bag, name)

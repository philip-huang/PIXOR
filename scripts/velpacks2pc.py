import roslib
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from sensor_msgs.msg import Image as ROSIMAGE
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from ros_numpy import point_cloud2 as rospc2

from helpers import *

bag = '/home/briallan/HawkEyeData/TestData/_2018-10-30-14-31-17'
if not os.path.exists(bag):
	os.mkdir(bag)

folder = bag + '/pc_data/'
if not os.path.exists(folder):
	os.mkdir(folder)

index = 0

h_metric = 40
w_metric = 30
meter_to_pixel = 20
bridge = CvBridge()
R = np.matrix([[ -0.006823, -0.999276, -0.037417 ], [-0.181251, 0.038034, -0.982701 ], [0.983413,  0.000077,  -0.181379]])
t = np.array([ 0.021637, -0.30, -0.920656 ])
T_cam_velo = np.eye(4)
T_cam_velo[0:3,0:3] = R
T_cam_velo[0:3,3] = t
#To-Do: create files for the timestamps for img, velo

timestamps = np.genfromtxt(bag+'/velo/velotimestamps.csv', delimiter=',')
toLook = timestamps[:,1] - timestamps[0][1]
#print(toLook)

def callbackROS(velo_ros):
  global index
  if index in toLook:
    arr = rospc2.pointcloud2_to_array(velo_ros)
    velo = get_xyzi_points(arr)

    #velo = passthrough(velo, 0, 70, -40, 40, -3.0, 3.5)
    intensities = velo[:,3]
    xyz = np.hstack((velo[:,0:3], np.ones((velo.shape[0], 1))))
    #xyz = xyz.dot(T_cam_velo.T)
    
    #t = np.copy(xyz)

    #xyz[:,0] = t[:,2]
    #xyz[:,1] = t[:,0]
    #xyz[:,2] = t[:,1]
    xyz[:,3] = intensities/255
    indexs = int(np.where(toLook==index)[0])
    indexs = str(indexs).zfill(4)
    xyz.reshape((xyz.size)) #one dimension
    xyz.astype(np.float32).tofile(folder + 'pc' + indexs + '.bin')
  index += 1
	#print('Index: ' + indexs + ' img: ' + str(img_ros.header.seq) + ' velo: ' + str(velo_ros.header.seq))


if __name__ == '__main__':
	rospy.init_node('convert2pc')
	vel_sub = message_filters.Subscriber('velodyne_points', PointCloud2)
	ts = message_filters.ApproximateTimeSynchronizer([vel_sub], queue_size=100, slop=0.07)
	ts.registerCallback(callbackROS)
	print('Listening...')
	rospy.spin()



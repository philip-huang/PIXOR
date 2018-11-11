#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


def is_point_in_ROI(x, y, z):
    """ Takes in a coordinate and determines if it is in front of car
        Args:
            x, y, z
            
        Returns:
            bool
    """
    xinbound = True if 0 <= x <= 70 else False
    yinbound = True if -40 <= y <= 40 else False
    zinbound = True if -2.5 <= z <= 1 else False   
 
    return xinbound and yinbound and zinbound

def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZI
        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message
            
        Returns:
            pcl.PointCloud_PointXYZI: PCL XYZI point cloud
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        # Remove points that lie outside of ROI
        if is_point_in_ROI(data[0], data[1], data[2]):
            points_list.append([data[0], data[1], data[2], data[3]])
    
    pcl_data = pcl.PointCloud_PointXYZI()
    pcl_data.from_list(points_list)

    return pcl_data

def callback_convert(data):
    # takes in velodyne_points and converts each frame to pcl
    pcl_data = ros_to_pcl(data)
    
        
    return
    
    
    
def listener():
    # listens to velodyne_points topic and processes frame
    rospy.init_node('packs2points', anonymous=True)

    rospy.Subscriber("/velodyne_points", PointCloud2, callback_convert)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    print(is_point_in_ROI(1,-0,1))

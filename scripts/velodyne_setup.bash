#! /bin/bash

echo "Setting up velodyne drivers."  

dir=$(pwd)

killall -9 rosmaster
roscore 


source $dir/../velodyne_ws/devel/setup.bash 
roslaunch velodyne_driver nodelet_manager.launch &

#rosrun cloud_node must take in absolute path of calibration file

rosrun velodyne_pointcloud cloud_node _calibration:=$dir/vel_64_calib.yaml &

rosrun image_transport republish compressed in:=/blackfly/image_color raw out:=/blackfly/image_color &


exit 0

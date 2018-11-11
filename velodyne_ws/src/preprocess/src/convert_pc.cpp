#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>


ros::Publisher pub;

std::vector<std::vector<std::vector<float> > >  preprocess_pc(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
  //x is 1.1, 40       0.1m   Matrix size is 390 x 291 x 66
  //y is -14.5, 14.5   -->    Add one z layer for intensity, and another layer to count nonzeros
  //z is -3.0, 3.5     -->    Final size use 390 x 291 x 68 (will cut away last layer)
  
  bool xrange, yrange, zrange;
   std::cout<<"h";
  std::vector<std::vector<std::vector<float> > > 
        output (390,std::vector<std::vector<float> >(291,std::vector <float>(68,0)));
 
  for (long i = 0; i <1000 ; i++){  //cloud.points.size ()
    int x = static_cast<int>((cloud->points[i].x - 1.1) * 10);
    int y = static_cast<int>((cloud->points[i].y + 14.5) * 10);
    int z = static_cast<int>((cloud->points[i].z + 3) * 10);
    xrange = x < 390;
    yrange = y < 291;
    zrange = z < 66;
    if(xrange && yrange && zrange){
      output[x][y][z] = 1;
      float cur_avg = output[x][y][66];
      int count = output[x][y][67];
      output[x][y][66] = (cur_avg * count + cloud->points[i].intensity)/(count+1);
      output[x][y][67] += 1;
    }
  }
  return output;
}


void pcCallback (const sensor_msgs::PointCloud2ConstPtr& input){
  //convert ROS message to point could
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*input,pcl_pc2);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromPCLPointCloud2(pcl_pc2,*cloud);

  //pcl::PointCloud<pcl::PointXYZI> cloud;
  //pcl::fromROSMsg(*input, cloud);
  //std::cout<<"h";
  preprocess_pc(cloud);



}

int main(int argc, char** argv){ 
  ros::init(argc, argv, "preprocess");
  ros::NodeHandle nh;
  std::cout<<"Listening to velodyne_points... "<<std::endl;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("velodyne_points", 1000, pcCallback);

  ros::spin ();
}

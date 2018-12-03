#include <stdio.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <iomanip>  
#include <vector>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <fstream>    
#include <string>
#include <vector> 

using namespace std;
extern "C"
{	

	const float x_MIN = 0.0;
	const float x_MAX = 70.0;
	const float y_MIN =-40.0;
	const float y_MAX = 40.0;
	const float z_MIN = -2.5; // -0.4	////TODO : to be determined ....
	const float z_MAX = 1; // 2
	const float x_DIVISION = 0.1;
	const float y_DIVISION = 0.1;
	const float z_DIVISION = 0.1;	// was 0.2 originally

	int X_SIZE = (int)((x_MAX-x_MIN)/x_DIVISION);	// X_SIZE is 700
	int Y_SIZE = (int)((y_MAX-y_MIN)/y_DIVISION);   // Y_SIZE is 800
	int Z_SIZE = (int)((z_MAX-z_MIN)/z_DIVISION);   // Z_SIZE is 35
    
	inline int getX(float x){
		return (int)((x-x_MIN)/x_DIVISION);
	}

	inline int getY(float y){
		return (int)((y-y_MIN)/y_DIVISION);
	}

	inline int getZ(float z){
		return (int)((z-z_MIN)/z_DIVISION);
	}
    
    inline int at3(int a, int b, int c){
        return a * (X_SIZE * (Z_SIZE+1)) + b * (Z_SIZE+1) + c;
    }    
    inline int at2(int a, int b){
        return a * X_SIZE+ b;
    }

    void createTopViewMaps(const void * indatav, const char* path)
    {	
        
	    float * data_cube = (float *) indatav;
		// load point cloud
		FILE *fp;
	  	
	  	// IMPORTANT: CHANGE THIS FILE PATH TO THE .bin VELODYNE FILES
		//string velo_dir  = "/mnt/ssd2/od/KITTI/training/velodyne/";
	
	    int32_t num = 1000000;
	    float *data = (float*)malloc(num*sizeof(float));

	    float *px = data+0;
		float *py = data+1;
		float *pz = data+2;
		float *pr = data+3;

		//ostringstream velo_filename;
		//velo_filename << setfill('0') << setw(6) << frame_counter << ".bin";

		//string velo_path = velo_dir + velo_filename.str();

		//const char* x = velo_path.c_str();
		
        fp = fopen (path, "rb");
        
		if(fp == NULL){
			cout << path << " not found. Ensure that the file path is correct." << endl;
		}

		num = fread(data,sizeof(float),num,fp)/4;
        //cout<<"number of points: "<<num<<endl;
		//3D grid box index
	
		//height features X_SIZE * Y_SIZE * (Z_SIZE + 1)
		//density feature X_SIZE * Y_SIZE * 1
		std::vector<int> density_map(Y_SIZE*X_SIZE);
        //float * occupancy_map = new float [Y_SIZE * X_SIZE * (Z_SIZE+1)];
 		
        
		for (int32_t i=0; i<num; i++) {

            //cout<<point.x<<" " <<point.y<<" "<<point.z<<" " <<point.intensity<<endl;		
			//For every point in each cloud, only select points inside a predefined 3D grid box
			if (*px > x_MIN && *py > y_MIN && *pz >z_MIN && *px < x_MAX && *py < y_MAX && *pz < z_MAX)
			{
                int X = getX(*px);
                int Y = getY(*py);
                int Z = getZ(*pz);
				//For every point in predefined 3D grid box..... 
                *(data_cube + at3(Y, X, Z)) = 1; 
			    *(data_cube + at3(Y, X, Z_SIZE)) += *pr;
				density_map[at2(Y, X)]++;	// update count#, need to be normalized afterwards	
			}
		    px+=4; py+=4; pz+=4; pr+=4;
		}
        
        for (int y = 0; y < Y_SIZE; y++) {
            for (int x = 0; x < X_SIZE; x++) {
                if (density_map[at2(y, x)] > 0){
                *(data_cube + at3(y, x, Z_SIZE)) = *(data_cube + at3(y, x, Z_SIZE)) / (float)density_map[at2(y, x)];
                }
            }
        } 
        //cout<<"Last step start"<<endl;
         
		fclose(fp);

		delete[] data;
        		
		// cout << "C++ part exited. LiDAR data pre-processing completed" << endl;
    }
}

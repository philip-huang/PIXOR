import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import cv2

rows = 800
cols = 700
height = 36
path = '/mnt/ssd2/od/KITTI/training/velodyne'

print ('LiDAR data pre-processing starting...')

# initialize an np 3D array with 1's
indata = np.zeros((rows, cols, height), dtype = np.float32)

# IMPORTANT: CHANGE THE FILE PATH TO THE .so FILE
# create a handle to LidarPreprocess.c
SharedLib = ctypes.cdll.LoadLibrary('./LidarPreprocess.so')

for frameNum in range(1):

	# call the C function to create top view maps
	# The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps 
    cdata = ctypes.c_void_p(indata.ctypes.data)
    apath = bytes(os.path.join(path, '000000.bin'), 'utf-8')
    
    tic = time.time()
    SharedLib.createTopViewMaps(cdata, apath)
    print("Time", time.time()-tic)
    
    check = np.load(str(os.path.join(path, '000000.npy')))
    print("diff", (check-indata).sum())    
	# At this point, the pre-processed current frame is stored in the variable indata which is a 400x400x8 array.

	# Pass indata to the rest of the MV3D pipeline.

	# Code to visualize the 8 top view maps (optional)
    np.save('1', indata)
    #cv2.imwrite('gt.png', check[:, :, -1])
    #cv2.imwrite('test.png', indata[:, : -1])
	# for i in range(8):
		
	# 	plt.subplot(2, 4, i+1)
	# 	plt.imshow(indata[:,:,i])
	# 	plt.gray()

	# plt.show()

print ('LiDAR data pre-processing complete for', frameNum + 1, 'frames')

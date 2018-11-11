## How to read labeled data

### Label Folder Structure
    .
    ├── (rosbag_name)
        ├── img                       # Folder of camera images
            ├── img0000.png           # Camera Image 0
            ├── img0001.png           # Camera Image 1...
          ....
        ├── labels                    # Folder of label Txt files
            ├──bev0000.txt            # Label for Frame 1
            ├──bev0001.txt            # Label for Frame 2
          ...
        ├── velo                      # Birds'Eye View Image of Lidar
            ├──bev0000.png            # Lidar BEV frame 1
            ├──bev0001.png            # Lidar BEV frame 2
            ├──bev0002.png
          ...
            ├──velotimestamps.csv     # Timestamp and sequences of all BEV frames

        ├── (rosbag_name).bag         # Actual Rosbag
    ├── plot_GT.py                    # Utility Script to plot ground truth labels
    ├── via2label.py                  # Utility Script to generate label txt files from VIA's ugly json file
    .
    
### Labeling tools Used
VGG Image Annotator (VIA) from Visual Geometry Group in University of Oxford
[Link] (https://www.robots.ox.ac.uk/~vgg/software/via/)

### Label Format

All labels are given in the **Velodyne frame** and in **metric** space

An object entry contains the Following:
```
Name, min_x, min_y, width , height
```
where minumum_x (y) is the smallest possible x(y) coordinate circled by the rectangular bounding box

Example:
```
Pedestrian 13.00 2.75 0.45 0.65
```

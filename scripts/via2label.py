from os.path import join
import json
import argparse

from helpers import *


parser = argparse.ArgumentParser()
parser.add_argument('data', help = "path to test data folder with all the seperate bag folders")
parser.add_argument('bag', help ="name of rosbag")


args = parser.parse_args()

bag = args.bag
data_path = args.data

json_path =  "../" + bag + '.json'  #look out here to change if necessary
labels = json.load(open(json_path))

'''
 x axis represents distance in front of car
 y axis represents left - right distance wrt car (left is positive, s.t it's right-hand system)
'''

meter_to_pixel = 20
BEV_H = 40
BEV_W = 30


annotations = [a for a in labels['_via_img_metadata'].values()]
print("Processing {} frames".format(len(annotations)))


for annot in annotations: 
    name = annot['filename']
    label_name = name[:-4] + '.txt'
    print(label_name)
    with open(join(data_path, bag, 'labels', label_name), 'w') as f:
        for r in annot['regions']:
            object = r['region_attributes']['object']  #here attribute name might be "type", "name", "object" be careful
            if object == 'Ca':
                object = 'Car'
            shape = r['shape_attributes']
            if shape['name'] == 'rect':
                x = BEV_H*meter_to_pixel - (shape['y']+shape['height'])
                y = BEV_W*meter_to_pixel/2 - (shape['x']+shape['width'])
                w = shape['width'] 
                h = shape['height'] 
                
                
                rect = [  [x  , y],  
                          [x+w, y], 
                          [x+w, y+h],  
                          [x,  y+h]  ]
                w, l, cx, cy, yaw = points2centroid_yaw(rect[0], rect[1], rect[2], rect[3])
                w /= meter_to_pixel
                l /= meter_to_pixel
                cx /= meter_to_pixel
                cy /= meter_to_pixel
                print('object {} at x={}m, y={}m, w={}m, h={}, yaw={:.0f}deg'.format(object, x, y, w, h, yaw * 180 / np.pi))
                f.write('{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(object, w, l, cx, cy,yaw))
 
            if shape['name'] == 'polyline':
                x = BEV_H*meter_to_pixel - np.array(shape['all_points_y'])
                y = BEV_W*meter_to_pixel/2 - np.array(shape['all_points_x'])
           
                points = np.array([[x[0], y[0]],    
                                   [x[1], y[1]],
                                   [x[2], y[2]],
                                   [x[3], y[3]]])
                
                rect = minimum_bounding_rectangle(points)
                w, l, cx, cy, yaw = points2centroid_yaw(rect[0], rect[1], rect[2], rect[3])
                w /= meter_to_pixel
                l /= meter_to_pixel
                cx /= meter_to_pixel
                cy /= meter_to_pixel
                print('object {} at x={:.1f}m, y={:.1f}m, w={:.1f}m, h={:.1f}, yaw={:.0f}deg'.format(object, cx, cy, w, l, yaw * 180 / np.pi))
                f.write('{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(object, w, l, cx, cy, yaw))
                







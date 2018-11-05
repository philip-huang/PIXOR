from os.path import join
import json

bag = '_2018-10-30-14-31-17'

label_path = 'label' + bag + '.json'
labels = json.load(open(label_path))

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
    with open(join(bag, 'labels', label_name), 'w') as f:
        for r in annot['regions']:
            object = r['region_attributes']['Name']
            
            shape = r['shape_attributes']
            if shape['name'] == 'rect':
                x = BEV_H - (shape['y']+shape['height']) / meter_to_pixel
                y = BEV_W/2 - (shape['x']+shape['width']) / meter_to_pixel
                w = shape['width'] / meter_to_pixel
                h = shape['height'] / meter_to_pixel
                
                print('object {} at x={}m, y={}m, w={}m, h={}'.format(object, x, y, w, h))
                
                f.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(object, x, y, w, h))
                
    
    

car = "Car 25.05 4.15 4.4 1.45 \n" \
    "Car 28.4 3.80 4.65 1.5 \n" \
    "Car 31.05 4.20 5.0 1.6"
    
for annot in annotations[1:]:
    name = annot['filename']
    label_name = name[:-4] + '.txt'
    with open(join(bag, 'labels', label_name), 'a') as f:
        f.write(car)



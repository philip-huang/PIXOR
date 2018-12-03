'''
Script to run PIXOR Detector on KITTI Raw Dataset
Generate a Series of BEV Predictions Images
'''
import torch
import numpy as np
import pykitti
import cv2

import os.path
import ctypes
import time

from model import PIXOR
from postprocess import filter_pred, non_max_suppression
from utils import get_bev, plot_bev


class Detector(object):

    def __init__(self, config, cdll):
        self.config = config
        self.cdll = cdll
        if self.cdll:
            self.LidarLib = ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so')
        #self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.net = PIXOR(config['geometry'], config['use_bn']).to(self.device)

        self.net.set_decode(True)
        self.net.load_state_dict(torch.load(config['ckpt_name'], map_location=self.device))
        self.net.eval()

        for p in self.net.parameters():
            p.require_grad = False

        print("PIXOR BEV Detector Initialized!")

    def preprocess(self, velo, path):
        geom = self.config['geometry']
        velo_processed = np.zeros(geom['input_shape'], dtype=np.float32)
        if self.cdll:
            c_name = bytes(path, 'utf-8')
            c_data = ctypes.c_void_p(velo_processed.ctypes.data)
            self.LidarLib.createTopViewMaps(c_data, c_name)
        else:
            def passthrough(velo):
                q = (geom['W1'] < velo[:, 0]) * (velo[:, 0] < geom['W2']) * \
                    (geom['L1'] < velo[:, 1]) * (velo[:, 1] < geom['L2']) * \
                    (geom['H1'] < velo[:, 2]) * (velo[:, 2] < geom['H2'])
                indices = np.where(q)[0]
                return velo[indices, :]

            velo = passthrough(velo)
            velo_processed = np.zeros(geom['input_shape'], dtype=np.float32)
            intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
            for i in range(velo.shape[0]):
                x = int((velo[i, 1] - geom['L1']) / 0.1)
                y = int((velo[i, 0] - geom['W1']) / 0.1)
                z = int((velo[i, 2] - geom['H1']) / 0.1)
                velo_processed[x, y, z] = 1
                velo_processed[x, y, -1] += velo[i, 3]
                intensity_map_count[x, y] += 1
            velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1], intensity_map_count, \
                                                 where=intensity_map_count != 0)

        velo_processed = torch.from_numpy(velo_processed).permute(2, 0, 1).to(self.device)
        velo_processed.require_grad=False
        return velo_processed

    def postprocess(self, pred):
        cls_pred = pred[..., 0]
        activation = cls_pred > self.config['cls_threshold']
        num_boxes = int(activation.sum())

        if num_boxes == 0:
            print("No bounding box found")
            return [], []

        corners = torch.zeros((num_boxes, 8))
        for i in range(1, 9):
            corners[:, i - 1] = torch.masked_select(pred[i, ...], activation)
        corners = corners.view(-1, 4, 2).numpy()
        scores = torch.masked_select(cls_pred, activation).cpu().numpy()

        # NMS
        selected_ids = non_max_suppression(corners, scores, self.config['nms_iou_threshold'])
        corners = corners[selected_ids]
        scores = scores[selected_ids]

        return corners, scores

    def __call__(self, velo, path):
        t_start = time.time()
        bev = self.preprocess(velo, path)
        t_pre = time.time()
        with torch.no_grad():
            pred = self.net(bev.unsqueeze(0)).squeeze_(0)

        t_m = time.time()
        corners, scores = filter_pred(self.config, pred)
        input_np = bev.permute(1, 2, 0).cpu().numpy()

        t_post = time.time()
        pred_bev = get_bev(input_np, corners)

        t_s = [t_pre-t_start, t_m-t_pre, t_post-t_m]
        return t_s, corners, scores, pred_bev

def run(dataset, save_path, height=400):
    config = {
      "ckpt_name": "experiments/decay/34epoch",
      "use_bn": True,
      "cls_threshold": 0.5,
      "nms_iou_threshold": 0.1,
      "nms_top": 64,
      "geometry": {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.0,
        'grid_size': 0.1,
        'input_shape': [800, 700, 36],
        'label_shape': [200, 175, 7],
        },
    }
    
    # Initialize Detector
    cdll = True
    pixor = Detector(config, cdll)
    
    # Initialize path to Kitti dataset
    
    # Video Writer from OpenCV
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Be sure to use lower case
    imshape = (config['geometry']['input_shape'][1] * 2, config['geometry']['input_shape'][0])
    videowriter = cv2.VideoWriter(save_path, fourcc, 10.0, imshape)

    avg_time = []
    for (item, velo_file) in enumerate(dataset.velo_files):
        velo = dataset.get_velo(item)
        path = dataset.velo_files[item]
        time, corners, scores, pred_bev = pixor(velo, path)
        #print(pred_bev.shape)
        print(time)
        merged_im = np.zeros((pred_bev.shape[0], pred_bev.shape[1] * 2, 3), dtype=np.uint8)
        avg_time.append(time) 
        image = np.asarray(dataset.get_cam2(item), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        rows, cols = image.shape[:2]
        crop_x = int(merged_im.shape[0]//2 - height//2)
        crop_y = int(merged_im.shape[0]//2 + height//2)
        image = cv2.resize(image, (int(height/rows * cols), height))
        #print(image.shape)
        crop_i = int(image.shape[1]//2 - merged_im.shape[1]/4)
        crop_j = int(image.shape[1]//2 + merged_im.shape[1]/4) 
        #print(crop_i, crop_j)
        merged_im[crop_x:crop_y, :pred_bev.shape[1], :] = image[:, crop_i:crop_j,:]
        merged_im[:, pred_bev.shape[1]:, :] = pred_bev
        videowriter.write(merged_im)        
    
    videowriter.release()
    avg_time = np.mean(avg_time, axis=0)
    print("Average Preprocessing Time:  {:.3f}s \n"
          "        Forward Time:        {:.3f}s \n"
          "        Postprocessing Time: {:.3f}s"
          .format(avg_time[0], avg_time[1], avg_time[2]))

def make_kitti_video():
     
    basedir = '/mnt/ssd2/od/KITTI/raw'
    date = '2011_09_26'
    drive = '0035'
    dataset = pykitti.raw(basedir, date, drive)
   
    videoname = "detection_{}_{}.avi".format(date, drive)
    save_path = os.path.join(basedir, date, "{}_drive_{}_sync".format(date, drive), videoname)    
    run(dataset, save_path)
     
def make_test_video():

    basedir = '/mnt/ssd2/od/testset'
    date = '_2018-10-30-15-25-07'
    import testset
    dataset = testset.TestSet(basedir, date)
    
    videoname = "detection_{}.avi".format(date)
    save_path = os.path.join(basedir, date, videoname)
    run(dataset, save_path, height=700)


if __name__ == '__main__':
    make_test_video()

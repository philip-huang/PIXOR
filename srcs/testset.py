import glob
import pandas as pd
import numpy as np
import os.path
from PIL import Image

class TestSet(object):
    def __init__(self, basedir, bagname):
        self.basedir = basedir
        self.bagname = bagname
        self.data_path = os.path.join(basedir, bagname)        
        self.imtype = 'png'        
        self._get_file_lists()        
        
    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam_files = sorted(glob.glob(
            os.path.join(self.data_path, 'img', '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.data_path, 'pc_data', '*.bin')))
    
        self.cam_stamps = pd.read_csv(os.path.join(self.data_path, 'img', 'imgtimestamps.csv'))
        self.velo_stamps = pd.read_csv(os.path.join(self.data_path, 'velo', 'velotimestamps.csv'))

    def get_cam2(self, idx):
        """Load an image from file."""
        mode = "RGB"
        return Image.open(self.cam_files[idx]).convert('RGB')
    
 
    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        scan = np.fromfile(self.velo_files[idx], dtype=np.float32)
        return scan.reshape((-1, 4))
    

def test():
    basedir = '/mnt/ssd2/od/testset'
    date = '_2018-10-30-15-25-07'
    dataset = TestSet(basedir, date)
    print(dataset.velo_files)
    im = dataset.get_cam2(0)    
    print(im.width)    
    print(im.height)
    scan = dataset.get_velo(0)
    np.set_printoptions(precision=3)
    print(scan.mean(axis=0))
    print(scan.max(axis=0))
    print(scan.min(axis=0))

if __name__=="__main__":
    test()

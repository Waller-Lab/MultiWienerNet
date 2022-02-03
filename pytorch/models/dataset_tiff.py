import skimage.io
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import torch
import numpy as np

class MiniscopeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, all_files,filepath_meas, transform=None):

        
        self.all_files_gt =  all_files
        self.filepath_meas = filepath_meas
#         self.all_files_gt = glob.glob(filepath_gt + '*.tiff')
        
        self.transform = transform

    def __len__(self):
        return len(self.all_files_gt)

    def __getitem__(self, idx):

        im_gt = skimage.io.imread(self.all_files_gt[idx])
        im_meas = skimage.io.imread(self.filepath_meas+self.all_files_gt[idx].split('/')[-1].split('.tiff')[0]+'.png')
        
        sample = {'im_gt': im_gt.astype('float32')/255., 'meas': im_meas.astype('float32')/255.}


        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    
    
class MiniscopeDataset_backup(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, filepath_meas, filepath_gt, transform=None):

        self.filepath_meas = filepath_meas
        self.filepath_gt =  filepath_gt
        self.all_files_gt = glob.glob(filepath_gt + '*.tiff')
        
        self.transform = transform

    def __len__(self):
        return len(self.all_files_gt)

    def __getitem__(self, idx):

        im_gt = skimage.io.imread(self.filepath_gt + str(idx) + '.tiff')
        im_meas = skimage.io.imread(self.filepath_meas + str(idx) + '.png')
        
        sample = {'im_gt': im_gt.astype('float32')/255., 'meas': im_meas.astype('float32')/255.}


        if self.transform:
            sample = self.transform(sample)

        return sample
class downsize(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, ds=0.5):
        self.ds=ds
        

    def __call__(self, sample):
        
        im_gt, meas = sample['im_gt'], sample['meas']
        meas=meas[18:466,4:644]
        meas= cv2.resize(meas, (0,0), fx=self.ds, fy=self.ds) 
        im_gt=im_gt[0:32,18:466,4:644]
        im_gt= cv2.resize(im_gt.transpose([1,2,0]), (0,0), fx=self.ds, fy=self.ds) 
        im_gt=im_gt.transpose([2,0,1])
        return {'im_gt': im_gt,
                'meas': meas}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        im_gt, meas = sample['im_gt'], sample['meas']

        return {'im_gt': torch.from_numpy(im_gt).unsqueeze(0),
                'meas': torch.from_numpy(meas).unsqueeze(0).unsqueeze(0)}
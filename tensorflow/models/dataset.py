import skimage.io
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MiniscopeDataset(Dataset):
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
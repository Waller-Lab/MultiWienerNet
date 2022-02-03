import numpy as np
import torch, torch.optim
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import os, sys, json, glob
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import matplotlib.pyplot as plt

import random

import skimage.io
import torch.nn as nn
import argparse

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import models.wiener_model as wm
import models.dataset as ds
from PIL import Image
import helper as hp

import scipy.io

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', default='2D')
parser.add_argument('--network', default='multiwiener') #'wiener' or 'unet' or 'multiwiener'
parser.add_argument('--id', default='') #some identifier
parser.add_argument('--loss_type', default='l1') 
parser.add_argument('--device', default='0') 
parser.add_argument('--psf_num', default=9, type=int)
parser.add_argument('--psf_ds', default=0.75, type=float)
parser.add_argument('--epochs', default=10000, type=int)
parser.add_argument('--lr', default=1e-4, type=float) 
parser.add_argument('--batch_size', default=1, type=int) 
parser.add_argument('--load_path',default=None)
parser.add_argument('--save_checkponts',default=True)

args = parser.parse_args()
#args = parser.parse_args(''.split())

os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# for 3D-UNet multiwiener
registered_psfs_path = '../data/multiWienerPSFStack_40z_aligned.mat'
psfs = scipy.io.loadmat(registered_psfs_path)
psfs=psfs['multiWienerPSFStack_40z']

if args.data_type == '3D':
    if args.network=='wiener' or args.network=='unet':
        psfs=hp.pre_process_psfs(psfs)[:,:,4]
        Ks=np.ones((32,1,1))
        print('choosing 1 psfs')

    elif args.network=='multiwiener':
        Ks=np.ones((args.psf_num,32,1,1))
        if args.psf_num==9:
            print('choosing 9 psfs')
            psfs=hp.pre_process_psfs(psfs)
    else:
        print('invalid network')
    psfs = hp.downsize_psf(psfs)
else: #2D
    if args.network=='wiener' or args.network=='unet':
        psfs=hp.pre_process_psfs_2d(psfs)[:,:,4, 0]
        Ks= 1.
        print('choosing 1 psfs')

    elif args.network=='multiwiener':
        Ks=np.ones((args.psf_num,1,1))
        if args.psf_num==9:
            print('choosing 9 psfs')
            psfs=hp.pre_process_psfs_2d(psfs)[...,0]
            psfs = psfs.transpose(2,0,1)
    else:
        print('invalid network')

    
down_size = ds.downsize(ds=args.psf_ds)
to_tensor = ds.ToTensor()
add_noise=ds.AddNoise()

if args.data_type == '3D':
    filepath_gt = '/home/kyrollos/LearnedMiniscope3D/Data3D/Training_data_all/' 
else:
    filepath_gt = '/home/kyrollos/LearnedMiniscope3D/Data/Target/'
    filepath_meas = '/home/kyrollos/LearnedMiniscope3D/Data/Train/'


filepath_all=glob.glob(filepath_gt+'*')
random.Random(8).shuffle(filepath_all)
print('total number of images',len(filepath_all))
total_num_images = len(filepath_all)
num_test = 0.2 # 20% test
filepath_train=filepath_all[0:int(total_num_images*(1-num_test))]
filepath_test=filepath_all[int(total_num_images*(1-num_test)):]

print('training images:', len(filepath_train), 
      'testing images:', len(filepath_test))

if args.data_type == '3D':
    dataset_train = ds.MiniscopeDataset(filepath_train, transform = transforms.Compose([down_size,add_noise,to_tensor]))
    dataset_test = ds.MiniscopeDataset(filepath_test, transform = transforms.Compose([down_size,add_noise,to_tensor]))
else:
    dataset_train = ds.MiniscopeDataset_2D(filepath_train, filepath_meas, transform = transforms.Compose([ds.crop2d(),ds.ToTensor2d()]))
    dataset_test = ds.MiniscopeDataset_2D(filepath_test, filepath_meas, transform = transforms.Compose([ds.crop2d(),ds.ToTensor2d()]))


dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=1)

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                        shuffle=False, num_workers=1)

device = 'cuda:0'

if args.data_type == '3D':
    from models.unet3d import Unet
    unet_model = Unet(n_channel_in=args.psf_num, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu').to(device)

    if args.network == 'multiwiener' or args.network == 'wiener':
        wiener_model=wm.WienerDeconvolution3D(psfs,Ks).to(device)
        model=wm.MyEnsemble(wiener_model,unet_model)
    else:
        model = unet_model
else: #2D
    from models.unet import Unet
    if args.network == 'multiwiener':
        num_in_channels = args.psf_num
    else:
        num_in_channels = 1
        
    
    #unet_model = Unet(n_channel_in=num_in_channels, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu').to(device)
    unet_model = Unet(n_channel_in=num_in_channels, n_channel_out=1, residual=True, down='conv', up='nearest', activation='relu').to(device)

    if args.network == 'multiwiener' or args.network == 'wiener':
        wiener_model=wm.WienerDeconvolution3D(psfs,Ks).to(device)
        model=wm.MyEnsemble2d(wiener_model,unet_model)
    else:
        model = unet_model

    
if args.load_path is not None:
    model.load_state_dict(torch.load('saved_data/'+args.load_path,map_location=torch.device(device)))
    print('loading saved model')


loss_fn = torch.nn.L1Loss()
ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

if args.save_checkponts == True:
    filepath_save = 'saved_data/' +"_".join((list(vars(args).values()))[0:5]) + "/"

    if not os.path.exists(filepath_save):
        os.makedirs(filepath_save)

    with open(filepath_save + 'args.json', 'w') as fp:
        json.dump(vars(args), fp)
        


best_loss=27e7

for itr in range(0,args.epochs):
    for i_batch, sample_batched in enumerate(dataloader_train):
        optimizer.zero_grad()
        #out = model(sample_batched['meas'].repeat(1,32,1,1)[...,18:466,4:644].unsqueeze(0).to(device))
        if args.network=='unet' and args.data_type == '3D':
            out = model(sample_batched['meas'].repeat(1,1,32,1,1).to(device))
        else:
            out = model(sample_batched['meas'].to(device))

        if args.loss_type=='l1':
            loss = loss_fn(out, sample_batched['im_gt'].to(device))
        else:
            #loss = loss_fn(out, sample_batched['im_gt'].to(device))+ (1- ms_ssim( out[0], sample_batched['im_gt'][0].to(device), data_range=1, size_average=False))
            
            loss = loss_fn(out, sample_batched['im_gt'].to(device)) + (1-ssim_loss(out, sample_batched['im_gt'].to(device)))
        loss.backward()
        optimizer.step()
        if i_batch %100 ==0:
            print('epoch: ', itr, ' batch: ', i_batch, ' loss: ', loss.item(), end='\r')

        #break 
    if args.data_type == '3D':
        out_np = np.max(out.detach().cpu().numpy()[0,0],0)
        gt_np = np.max(sample_batched['im_gt'].detach().cpu().numpy()[0,0],0)
        meas_np = np.max(sample_batched['meas'].detach().cpu().numpy()[0,0],0)
    else:
        out_np = out.detach().cpu().numpy()[0][0]
        gt_np = sample_batched['im_gt'].detach().cpu().numpy()[0][0]
        meas_np = sample_batched['meas'].detach().cpu().numpy()[0][0]


    if args.save_checkponts == True:
        torch.save(model.state_dict(), filepath_save + 'model_noval.pt')
    
    if itr%1==0:
        total_loss=0
        for i_batch, sample_batched in enumerate(dataloader_test):
            with torch.no_grad():
                if args.network=='unet' and args.data_type == '3D':
                    out = model(sample_batched['meas'].repeat(1,1,32,1,1).to(device))
                else:
                    out = model(sample_batched['meas'].to(device))
                if args.loss_type=='l1':
                    loss = loss_fn(out, sample_batched['im_gt'].to(device))
                else:
                    loss = loss_fn(out, sample_batched['im_gt'].to(device)) + (1-ssim_loss(out, sample_batched['im_gt'].to(device)))
        
                    #loss = loss_fn(out, sample_batched['im_gt'].to(device))+(1- ms_ssim( out, sample_batched['im_gt'][0].to(device), data_range=1, size_average=False))
                
                
                total_loss+=loss.item()
                
        print('loss for testing set ',itr,' ',i_batch, total_loss)
                
            #break
        
        if args.save_checkponts == True:
            im_gt = Image.fromarray((np.clip(gt_np/np.max(gt_np),0,1)*255).astype(np.uint8))
            im = Image.fromarray((np.clip(out_np/np.max(out_np),0,1)*255).astype(np.uint8))
            im.save(filepath_save + str(itr) + '.png')
            im_gt.save(filepath_save + 'gt.png')
        
        
        if total_loss<best_loss:
            best_loss=total_loss

            # save checkpoint
            if args.save_checkponts == True:
                torch.save(model.state_dict(), filepath_save + 'model.pt')

        

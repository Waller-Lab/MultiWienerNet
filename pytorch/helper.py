import argparse, json, math
import scipy.io
import numpy as np
import cv2
#import models.wiener_model as wm
import torch

def to_np(x):
    x = x.detach().cpu().numpy()[0,0]
    x=x/np.max(x)
    return x

def max_proj(x, axis = 0):
    return np.max(x,axis)

def mean_proj(x, axis = 0):
    return np.mean(x,axis)

def calc_psnr(Iin,Itarget):
    
    mse=np.mean(np.square(Iin-Itarget))
    return 10*math.log10(1/mse)

def load_saved_args(model_file_path):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_type', default='3D')
    parser.add_argument('--num_images', default='multi') #'single' or 'multi
    parser.add_argument('--network', default='combined') #'combined' or 'unet'
    parser.add_argument('--device', default='0') 
    parser.add_argument('--epochs', default=10000, type=int)

    args = parser.parse_args("--device 1".split())

    with open(model_file_path+'args.json', "r") as f:
        args.__dict__=json.load(f)
    return args

def initialize_psfs(model_file_path):
    
    args = load_saved_args(model_file_path)
    # for 3D-UNet multiwiener
    registered_psfs_path = '../data/multiWienerPSFStack_40z_aligned.mat'
    psfs = scipy.io.loadmat(registered_psfs_path)
    psfs=psfs['multiWienerPSFStack_40z']
    if args.data_type == '3D':
        if args.network=='wiener' or args.network=='unet':
            psfs=psfs[18:466,4:644,4,0:32]
            Ks=np.ones((32,1,1))

        elif args.network=='multiwiener':
            Ks=np.ones((args.psf_num,32,1,1))
            if args.psf_num==9:
                print('choosing 9 psfs')
                psfs=psfs[18:466,4:644,:,0:32]

            elif args.psf_num==4:
                print('choosing 4 psfs')
                psfs=psfs[18:466,4:644,2:6,0:32]
            else:
                print('invalid psf num')
                
        psfs_ds=np.zeros((int(psfs.shape[0]*args.psf_ds),int(psfs.shape[1]*args.psf_ds),*psfs.shape[2:]))

        if args.psf_num>1:
            for p in range(psfs.shape[2]): 
                psfs_ds[:,:,p,:]=cv2.resize(psfs[:,:,p], (0,0), fx=args.psf_ds, fy=args.psf_ds)
            psfs_ds=np.transpose(psfs_ds,[2,3,0,1])

        else:
            psfs_ds=cv2.resize(psfs, (0,0), fx=args.psf_ds, fy=args.psf_ds)

            psfs_ds=np.transpose(psfs_ds,[2,0,1])

        psfs_ds=psfs_ds/np.max(psfs_ds)
        psfs=psfs_ds
    
    else: #2D
        if args.network=='wiener' or args.network=='unet':
            psfs=pre_process_psfs_2d(psfs)[:,:,4, 0]
            Ks= 1.
            print('choosing 1 psfs')

        elif args.network=='multiwiener':
            Ks=np.ones((args.psf_num,1,1))
            if args.psf_num==9:
                print('choosing 9 psfs')
                psfs=pre_process_psfs_2d(psfs)[...,0]
                psfs = psfs.transpose(2,0,1)
        else:
            print('invalid network')
    

        psfs = psfs/np.max(psfs)
    
    return psfs,Ks, args

def load_pretrained_model(filepath, model_type = 'unet', device = 'cuda:0'):
    from models.unet3d import Unet
    import models.wiener_model as wm
    if model_type == 'unet':
        model = Unet(n_channel_in=1, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu').to(device)

    elif model_type == 'wiener':
        psfs_wiener,Ks_wiener, args_wiener = initialize_psfs(filepath)
        unet_model = Unet(n_channel_in=1, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu').to(device)
        wiener_stage=wm.WienerDeconvolution3D(psfs_wiener,Ks_wiener).to(device)
        model=wm.MyEnsemble(wiener_stage,unet_model)

    elif model_type == 'multiwiener':
        psfs_multiwiener,Ks_multiwiener, args_multi= initialize_psfs(filepath)
        unet_stage = Unet(n_channel_in=9, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu').to(device)
        multiwiener_stage=wm.WienerDeconvolution3D(psfs_multiwiener,Ks_multiwiener).to(device)
        model=wm.MyEnsemble(multiwiener_stage,unet_stage)

    model.load_state_dict(torch.load(filepath+'model.pt',map_location=torch.device(device)))
    return model

def load_pretrained_model_2d(filepath, model_type = 'unet', device = 'cuda:0', load_model = True):
    from models.unet import Unet
    import models.wiener_model as wm
    if model_type == 'unet':
        model = Unet(n_channel_in=1, n_channel_out=1).to(device)#, residual=False, down='conv', up='nearest', activation='relu').to(device)

    elif model_type == 'wiener':
        psfs_wiener,Ks_wiener, args_wiener = initialize_psfs(filepath)
        #unet_model = Unet(n_channel_in=1, n_channel_out=1).to(device)#, residual=False, down='conv', up='nearest',activation='relu').to(device)
        
        unet_model = Unet(n_channel_in=1, n_channel_out=1, residual=True, down='conv', up='nearest',activation='relu').to(device)
        wiener_stage=wm.WienerDeconvolution3D(psfs_wiener,Ks_wiener).to(device)
        model=wm.MyEnsemble2d(wiener_stage,unet_model)

    elif model_type == 'multiwiener':
        psfs_multiwiener,Ks_multiwiener, args_multi= initialize_psfs(filepath)
        #unet_stage = Unet(n_channel_in=9, n_channel_out=1).to(device)#, residual=False, down='conv', up='nearest',activation='relu').to(device)
        unet_stage = Unet(n_channel_in=9, n_channel_out=1, residual=True, down='conv', up='nearest',activation='relu').to(device)
        multiwiener_stage=wm.WienerDeconvolution3D(psfs_multiwiener,Ks_multiwiener).to(device)
        model=wm.MyEnsemble2d(multiwiener_stage,unet_stage)

    if load_model == True:
        model.load_state_dict(torch.load(filepath+'model.pt',map_location=torch.device(device)))
    return model    
    
def pre_process_psfs(x):
    # Use this to make the image size a power of 2 for the network. 
    # Change these numbers according to your image size
    x = x[18:466,4:644,:,0:32]
    return x
def pre_process_psfs_2d(x):
    # Use this to make the image size a power of 2 for the network. 
    # Change these numbers according to your image size
    x = x[18:466,4:644]
    return x

def downsize_psf(psfs):
    psfs_ds=np.zeros((int(psfs.shape[0]*args.psf_ds),int(psfs.shape[1]*args.psf_ds),*psfs.shape[2:]))
    if args.psf_num>1:
        for p in range(psfs.shape[2]): 
            psfs_ds[:,:,p,:]=cv2.resize(psfs[:,:,p], (0,0), fx=args.psf_ds, fy=args.psf_ds)
        psfs_ds=np.transpose(psfs_ds,[2,3,0,1])
    else:
        psfs_ds=cv2.resize(psfs, (0,0), fx=args.psf_ds, fy=args.psf_ds)
        psfs_ds=np.transpose(psfs_ds,[2,0,1])

    psfs_ds=psfs_ds/np.max(psfs_ds)
    return psfs_ds
import torch.nn as nn
import torch
import numpy as np

class MultiWienerDeconvolution3D(nn.Module):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolution3D, self).__init__()
        initial_psfs = torch.tensor(initial_psfs, dtype=torch.float32)
        initial_Ks = torch.tensor(initial_Ks, dtype=torch.float32)

        self.psfs = nn.Parameter(initial_psfs, requires_grad =True)
        self.Ks = nn.Parameter(initial_Ks, requires_grad =True) #NEEED RELU CONSTRAINT HERE K is constrained to be nonnegative
        
    def forward(self, y):
        # Y preprocessing, Y is shape (N, C,H, W)
        h, w = y.shape[-3:-1]
        y = y.type(torch.complex64)

    
        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))

        # Temporarily transpose y since we cannot specify axes for fft2d
        Y=torch.fft.fft2(y)

        # Components preprocessing, psfs is shape (C,H, W)
        psf = self.psfs.type(torch.complex64)
        h_psf, w_psf = self.psfs.shape[0:2]

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0))

        H_sum = torch.fft.fft2(self.psfs)

        X=(torch.conj(H_sum)*Y)/ (torch.square(torch.abs(H_sum))+100*self.Ks)#, dtype=tf.complex64)
    
        x=torch.real((torch.fft.ifftshift(torch.fft.ifft2(X), dim=(-2, -1))))
        

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config
    
    
class WienerDeconvolution3D(nn.Module):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(WienerDeconvolution3D, self).__init__()
        initial_psfs = torch.tensor(initial_psfs, dtype=torch.float32)
        initial_Ks = torch.tensor(initial_Ks, dtype=torch.float32)

        self.psfs = nn.Parameter(initial_psfs, requires_grad =True)
        self.Ks = nn.Parameter(initial_Ks, requires_grad =True) #NEEED RELU CONSTRAINT HERE K is constrained to be nonnegative
        
    def forward(self, y):
        # Y preprocessing, Y is shape (N, C,H, W)
        h, w = y.shape[-3:-1]
        y = y.type(torch.complex64)

    
        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))

        # Temporarily transpose y since we cannot specify axes for fft2d
        Y=torch.fft.fft2(y)

        # Components preprocessing, psfs is shape (C,H, W)
        psf = self.psfs.type(torch.complex64)
        h_psf, w_psf = self.psfs.shape[0:2]

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0))

        H_sum = torch.fft.fft2(self.psfs)

        #print(H_sum.shape, Y.shape, self.Ks.shape)
        X=(torch.conj(H_sum)*Y)/ (torch.square(torch.abs(H_sum))+100*self.Ks)#, dtype=tf.complex64)
    
        x=torch.real((torch.fft.ifftshift(torch.fft.ifft2(X), dim=(-2, -1))))
        

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config    
    
    
class MyEnsemble2d(nn.Module):
    def __init__(self, wiener_model, unet_model):
        super(MyEnsemble2d, self).__init__()
        self.wiener_model = wiener_model
        self.unet_model = unet_model
    def forward(self, x):
        wiener_output = self.wiener_model(x)
        wiener_output = wiener_output/torch.max(wiener_output)
        final_output = self.unet_model(wiener_output)
        return final_output

class MyEnsemble(nn.Module):
    def __init__(self, wiener_model, unet_model):
        super(MyEnsemble, self).__init__()
        self.wiener_model = wiener_model
        self.unet_model = unet_model
    def forward(self, x):
        wiener_output = self.wiener_model(x)
        final_output = self.unet_model(wiener_output)
        return final_output
import tensorflow as tf
from tensorflow.keras import layers
from utils import *

class MultiWienerDeconvolution(layers.Layer):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolution, self).__init__()
        initial_psfs = tf.dtypes.cast(initial_psfs, dtype=tf.float32)
        initial_Ks = tf.dtypes.cast(initial_Ks, dtype=tf.float32)

        self.psfs = tf.Variable(initial_value=initial_psfs, trainable=True)
        self.Ks = tf.Variable(initial_value=initial_Ks, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, C)
        y_im=y
        _, h, w, _ = y.shape
        y = tf.dtypes.cast(y, dtype=tf.complex64)
        

        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))
        y = tf.pad(y, paddings=padding)

        # Temporarily transpose y since we cannot specify axes for fft2d
        y = tf.transpose(y, perm=[0, 3, 1, 2])   # Y is now shape (N, C, H, W)
        Y=tf.signal.fft2d(y)

        # Components preprocessing, psfs is shape (H, W, C)
        psf = tf.dtypes.cast(self.psfs, dtype=tf.complex64)
        h_psf, w_psf, _ = psf.shape

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0))

        H_sum = tf.pad(psf, paddings=padding_psf)

        H_sum = tf.transpose(H_sum, perm=[2, 0, 1])   # H_sum is now shape (C, H, W)
        H_sum = tf.signal.fft2d(H_sum)
        
        Ks = tf.transpose(self.Ks, [2, 0, 1]) # Ks is now shape (C, 1, 1)

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        # x goes from shape (N, C, H, W) -> (N, H, W, C)
        x = tf.transpose(x, [0, 2, 3, 1])

        x = crop_2d_tf(x)
#         x = tf.concat([x,y_im],-1)

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config
        

class MultiWienerDeconvolutionWFourier(layers.Layer):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolutionWFourier, self).__init__()
        initial_psfs = tf.dtypes.cast(initial_psfs, dtype=tf.float32)
        initial_Ks = tf.dtypes.cast(initial_Ks, dtype=tf.float32)

        self.psfs = tf.Variable(initial_value=initial_psfs, trainable=True)
        self.Ks = tf.Variable(initial_value=initial_Ks, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative
        self.fourier_weights = tf.Variable(initial_value=tf.ones([initial_psfs.shape[0],initial_psfs.shape[1],1]), trainable=True)
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, C)
#         y_im=y
        _, h, w, _ = y.shape
        y = tf.dtypes.cast(y, dtype=tf.complex64)
        

        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))
        y_fourier=tf.transpose(y, perm=[0, 3, 1, 2])
        y = tf.pad(y, paddings=padding)
        
        # Temporarily transpose y since we cannot specify axes for fft2d
        y = tf.transpose(y, perm=[0, 3, 1, 2])   # Y is now shape (N, C, H, W)
        Y=tf.signal.fft2d(y)
        Y_fourier=tf.signal.fft2d(y_fourier)
        

        
        
        # Components preprocessing, psfs is shape (H, W, C)
        psf = tf.dtypes.cast(self.psfs, dtype=tf.complex64)
        h_psf, w_psf, _ = psf.shape

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0))

        H_sum = tf.pad(psf, paddings=padding_psf)

        H_sum = tf.transpose(H_sum, perm=[2, 0, 1])   # H_sum is now shape (C, H, W)
        H_sum = tf.signal.fft2d(H_sum)
        
        Ks = tf.transpose(self.Ks, [2, 0, 1]) # Ks is now shape (C, 1, 1)

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
         
        #fourier map
        
        fourier_weights = tf.transpose(self.fourier_weights, perm=[2,0, 1]) 
        fourier_weights = tf.dtypes.cast(fourier_weights, dtype=tf.complex64)
        fourier_map=tf.math.multiply(Y_fourier,fourier_weights)
#         fourier_inv=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(fourier_map), axes=(2, 3))))
        fourier_inv=tf.math.real(tf.signal.ifft2d(fourier_map))
        
        fourier_inv=tf.transpose(fourier_inv, [0, 2, 3, 1])
#         print(fourier_inv.shape)
        
        
#         x goes from shape (N, C, H, W) -> (N, H, W, C)
        x = tf.transpose(x, [0, 2, 3, 1])

        x = crop_2d_tf(x)
#         print(x.shape)
        x = tf.concat([x,fourier_inv],-1)

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config        
        
class WienerDeconvolution(layers.Layer):
    """
    Performs Wiener Deconvolution in frequency domain.
    PSF, K are learnable parameters. K is enforced to be nonnegative everywhere.
    
    Input: initial_psf of shape (Y, X), initial_K is a scalar.
    """
    def __init__(self, initial_psf, initial_K):
#     def __init__(self):

        super(WienerDeconvolution, self).__init__()
        initial_psf = tf.dtypes.cast(initial_psf, dtype=tf.float32)
        initial_K = tf.dtypes.cast(initial_K, dtype=tf.float32)
        
        self.psf = tf.Variable(initial_value=initial_psf, trainable=True)
        self.K = tf.Variable(initial_value=initial_K, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, C)
        y_im=y
        
        _, h, w, _ = y.shape
        y = tf.squeeze(tf.dtypes.cast(y, dtype=tf.complex64), axis=-1) # Remove channel dimension
        
        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))))
        y = tf.pad(y, paddings=padding)
        Y=tf.signal.fft2d(y)

        # PSF preprocessing, psf is shape (H, W)
        psf = tf.dtypes.cast(self.psf, dtype=tf.complex64)
        h_psf, w_psf = psf.shape
        
        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))))
        H_sum = tf.pad(psf, paddings=padding_psf)
        H_sum=tf.signal.fft2d(H_sum)
        
        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*self.K, dtype=tf.complex64)
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(1, 2))))
                
        x = crop_2d_tf(x)

        x=x[..., None] # Add channel dimension
#         x = tf.concat([x,y_im],-1)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psf': self.psf.numpy(),
            'initial_K': self.K.numpy()
        })
        return config
    
    
    
class WienerDeconvolution3D(layers.Layer):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(WienerDeconvolution3D, self).__init__()
        initial_psfs = tf.dtypes.cast(initial_psfs, dtype=tf.float32)
        initial_Ks = tf.dtypes.cast(initial_Ks, dtype=tf.float32)

        self.psfs = tf.Variable(initial_value=initial_psfs, trainable=True)
        self.Ks = tf.Variable(initial_value=initial_Ks, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, C)
#         y_im=y
        _, h, w, _ = y.shape
        y = tf.dtypes.cast(y, dtype=tf.complex64)
        

        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))
        y = tf.pad(y, paddings=padding)

        # Temporarily transpose y since we cannot specify axes for fft2d
        y = tf.transpose(y, perm=[0, 3, 1, 2])   # Y is now shape (N, C, H, W)
        Y=tf.signal.fft2d(y)

        # Components preprocessing, psfs is shape (H, W, C)
        psf = tf.dtypes.cast(self.psfs, dtype=tf.complex64)
        h_psf, w_psf, _ = psf.shape

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0))

        H_sum = tf.pad(psf, paddings=padding_psf)

        H_sum = tf.transpose(H_sum, perm=[2, 0, 1])   # H_sum is now shape (C, H, W)
        H_sum = tf.signal.fft2d(H_sum)
        
        Ks = tf.transpose(self.Ks, [2, 0, 1]) # Ks is now shape (C, 1, 1)

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        # x goes from shape (N, C, H, W) -> (N, H, W, C)
        x = tf.transpose(x, [0, 2, 3, 1])

        x = crop_2d_tf(x)
#         x = tf.concat([x,y_im],-1)

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config
    
class MultiWienerDeconvolution3D(layers.Layer):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolution3D, self).__init__()
        initial_psfs = tf.dtypes.cast(initial_psfs, dtype=tf.float32)
        initial_Ks = tf.dtypes.cast(initial_Ks, dtype=tf.float32)

        self.psfs = tf.Variable(initial_value=initial_psfs, trainable=True)
        self.Ks = tf.Variable(initial_value=initial_Ks, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, C)
#         print(y.shape)
        y=tf.image.resize(y, [y.shape[1]//2,y.shape[2]//2])
        _, h, w, _ = y.shape
        y = tf.dtypes.cast(y, dtype=tf.complex64)
        
#         print(y.shape)
        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))
        y = tf.pad(y, paddings=padding)

        # Temporarily transpose y since we cannot specify axes for fft2d
        y = tf.transpose(y, perm=[0, 3, 1, 2])   # Y is now shape (N, C, H, W)
        Y=tf.signal.fft2d(y)

        # Components preprocessing, psfs is shape (H, W, C)
        psf = tf.dtypes.cast(self.psfs, dtype=tf.complex64)
        h_psf, w_psf, _,_ = psf.shape

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0),(0,0))
        

        H_sum = tf.pad(psf, paddings=padding_psf)
        

        H_sum = tf.transpose(H_sum, perm=[3,2, 0, 1])   # H_sum is now shape (C, H, W)
        H_sum_all = tf.signal.fft2d(H_sum)
        
        Ks_all = tf.transpose(self.Ks, [3,2, 0, 1]) # Ks is now shape (C, 1, 1)
        
        ##for one xy location
        H_sum=H_sum_all[:,0,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,0,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x1=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))

        ##for one xy location
        H_sum=H_sum_all[:,1,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,1,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x2=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        ##for one xy location
        H_sum=H_sum_all[:,2,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,2,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x3=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        ##for one xy location
        H_sum=H_sum_all[:,3,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,3,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x4=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        ##for one xy location
        H_sum=H_sum_all[:,4,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,4,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x5=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        ##for one xy location
        H_sum=H_sum_all[:,5,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,5,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x6=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        ##for one xy location
        H_sum=H_sum_all[:,6,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,6,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x7=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        ##for one xy location
        H_sum=H_sum_all[:,7,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,7,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x8=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        ##for one xy location
        H_sum=H_sum_all[:,8,:,:]#tf.expand_dims(H_sum_all[:,0,:,:],0)
        Ks=Ks_all[:,8,:,:]#tf.expand_dims(Ks_all[:,0,:,:],0)
        

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x9=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        # x goes from shape (N, C, H, W) -> (N, H, W, C)
        x=tf.concat([x1,x2,x3,x4,x5,x6,x7,x8,x9],1)
        x = tf.transpose(x, [0, 2, 3, 1])
        x = crop_2d_tf(x)
#         x = tf.concat([x,y_im],-1)

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config     
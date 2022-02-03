import numpy as np
import scipy.io
def pad2d (x):
    Ny=x.shape[0]
    Nx=x.shape[1]
    return np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2)), mode = 'constant')#, constant_values=(0))

def pad4d(x):
    Ny=x.shape[0]
    Nx=x.shape[1]
    return np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2),(0,0),(0,0)), mode = 'constant')#, constant_values=(0))

def crop4d(x,rcL,rcU,ccL,ccU):
    return x[rcL:rcU,ccL:ccU,:,:]

def crop2d(x,rcL,rcU,ccL,ccU):   
    return x[rcL:rcU,ccL:ccU]

def nocrop(x):
    return x

def nopad(x):
    return x

def A_2d_svd(x,H,weights,pad,mode='shift_variant'): #NOTE, H is already padded outside to save memory
    x=pad(x)
    #Y=np.zeros((x.shape[0],x.shape[1]))
    Y=np.zeros_like(x)
        
    if (mode =='shift_variant'):
        for r in range (0,weights.shape[2]):
            X=np.fft.fft2((np.multiply(pad(weights[:,:,r]),x)))
            Y=Y+ np.multiply(X,H[:,:,r])
    
    return np.real((np.fft.ifftshift(np.fft.ifft2(Y))))

def A_2d_svd_power(x,H,weights,pad,mode='shift_variant'): #NOTE, H is already padded outside to save memory
    Y=np.zeros_like(x)
        
    if (mode =='shift_variant'):
        for r in range (0,weights.shape[2]):
            X=np.fft.fft2((np.multiply(pad(weights[:,:,r]),x)))
            Y=Y+ X*H[:,:,r]
    
    return np.real((np.fft.ifftshift(np.fft.ifft2(Y))))

def A_2d_svd_crop(x,H,weights,pad,crop_indices,mode='shift_variant'): #NOTE, H is already padded outside to save memory
    x=pad(x)
    #Y=np.zeros((x.shape[0],x.shape[1]))
    Y=np.zeros_like(x)
        
    if (mode =='shift_variant'):
        for r in range (0,weights.shape[2]):
            X=np.fft.fft2((np.multiply(pad(weights[:,:,r]),x)))
            Y=Y+ np.multiply(X,H[:,:,r])
    
    return crop2d(np.real((np.fft.ifftshift(np.fft.ifft2(Y)))),*crop_indices)

def A_2d(x,psf,pad):
    X=np.fft.fft2((pad(x)))
    H=np.fft.fft2((pad(psf)))
    Y=np.multiply(X,H)
    
    return np.real((np.fft.ifftshift(np.fft.ifft2(Y))))

def A_2d_adj_svd(Hconj,weights,y,pad):
    y=pad(y)
    x=np.zeros_like(y)
    #x=np.zeros((y.shape[0],y.shape[1]))
    for r in range (0, weights.shape[2]):
        x=x+np.multiply(pad(weights[:,:,r]),(np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(Hconj[:,:,r], np.fft.fft2((y))))))))
    #note the weights are real so we dont take the complex conjugate of it, which is the adjoint of the diag 
    return x

def A_2d_adj(y,psf,pad):
    H=np.fft.fft2((pad(psf)))
    Hconj=np.conj(H)
    x=(np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(Hconj, np.fft.fft2((pad(y))))))))
    
    return x

def A_3d(x,h,pad):
    #h is the psf stack
    #x is the variable to convolve with h
    x=pad(x)
    B=np.zeros((x.shape[0],x.shape[1]))
        

    for z in range (0,h.shape[2]):
        #X=np.fft.fft2((np.multiply(pad(weights[:,:,z]),x)))
        B=B+ np.multiply(np.fft.fft2(x[:,:,z]),np.fft.fft2(pad(h[:,:,z])))
    
    return np.real((np.fft.ifftshift(np.fft.ifft2(B))))

def A_3d_svd_power(v,H,weights,pad):
    #alpha is Ny-Nx-Nz-Nr, weights
    #v is Ny-Nx-Nz
    #H is Ny-Nx-Nz-Nr
    # b= sum_r (sum_z (h**alpra.*v))
    #b=np.zeros((v.shape[0],v.shape[1]))
    b=np.zeros_like(v)
    for r in range (H.shape[3]):
        for z in range (H.shape[2]):
            b=b+np.multiply(H[:,:,z,r],np.fft.fft2(np.multiply(v,pad(weights[:,:,z,r]))))
    
    return np.real(np.fft.ifftshift(np.fft.ifft2(b)))

def A_3d_svd_crop(v,H,weights,pad, crop_indices):
    #alpha is Ny-Nx-Nz-Nr, weights
    #v is Ny-Nx-Nz
    #H is Ny-Nx-Nz-Nr
    # b= sum_r (sum_z (h**alpra.*v))
    b=np.zeros_like(v[:,:,0])
    #b=np.zeros((v.shape[0],v.shape[1]))
    for r in range (H.shape[3]):
        for z in range (H.shape[2]):
            b=b+np.multiply(H[:,:,z,r],np.fft.fft2(np.multiply(v[:,:,z],pad(weights[:,:,z,r]))))
    
    return crop2d(np.real(np.fft.ifftshift(np.fft.ifft2(b))),*crop_indices)

def A_3d_svd(v,alpha,H,pad):
    #alpha is Ny-Nx-Nz-Nr, weights
    #v is Ny-Nx-Nz
    #H is Ny-Nx-Nz-Nr
    # b= sum_r (sum_z (h**alpra.*v))
    b=np.zeros((v.shape[0],v.shape[1]))
    for r in range (H.shape[3]):
        for z in range (H.shape[2]):
            b=b+np.multiply(H[:,:,z,r],np.fft.fft2(np.multiply(v[:,:,z],alpha[:,:,z,r])))
    
    return np.real(np.fft.ifftshift(np.fft.ifft2(b)))



def A_3d_adj(x,h,pad):
    y=np.zeros_like(h)
    X=np.fft.fft2(pad(x))
    for z in range(h.shape[2]):
        H=np.conj(np.fft.fft2(pad(h[:,:,z])))
        y[:,:,z]=np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(H,X))))
    return y

# def A_2d_adj_svd(Hconj,weights,y,pad):
#     y=pad(y)
#     x=np.zeros_like(y)
#     #x=np.zeros((y.shape[0],y.shape[1]))
#     for r in range (0, weights.shape[2]):
#         x=x+np.multiply(pad(weights[:,:,r]),(np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(Hconj[:,:,r], np.fft.fft2((y))))))))
#     #note the weights are real so we dont take the complex conjugate of it, which is the adjoint of the diag 
#     return x

#def A_3d_adj_svd(b,alpha,Hconj,pad):
def A_3d_adj_svd(Hconj,alpha,x,pad):
    #y=sum_r(alpha.*H_conj**b)
    y=np.zeros((Hconj.shape[0],Hconj.shape[1],Hconj.shape[2])) 
    #y = pad(y)
    B=np.fft.fft2(pad(x))
    for z in range(alpha.shape[2]):
        for r in range(alpha.shape[3]):
            y[:,:,z]=y[:,:,z]+ pad(alpha[:,:,z,r])* np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(B,Hconj[:,:,z,r]))))
        
    return y

def grad(v):
    return np.array(np.gradient(v))  #returns gradient in x and in y


def grad_adj(v):  #adj of gradient is negative divergence
    z = np.zeros((n,n)) + 1j
    z -= np.gradient(v[0,:,:])[0]
    z -= np.gradient(v[1,:,:])[1]
    return z

def sim_data(im,H,weights,crop_indices):
    mu=0
    sigma=np.random.rand(1)*0.02+0.005 #abit much maybe 0.04 best0.04+0.01
    PEAK=np.random.rand(1)*1000+50

    I=im/np.max(im)
    #I[I<0.12]=0
    sim=crop2d(A_2d_svd(I,H,weights,pad2d),*crop_indices)
    sim=sim/np.max(sim)
    sim=np.maximum(sim,0)

    p_noise = np.random.poisson(sim * PEAK)/PEAK

    g_noise= np.random.normal(mu, sigma, 648*486)
    g_noise=np.reshape(g_noise,(486,648))
    sim=sim+g_noise+p_noise
    sim=sim/np.max(sim)
    sim=np.maximum(sim,0)
    sim=sim/np.max(sim)
    return sim


# load in forward model weights
def load_weights():
    h=scipy.io.loadmat('/home/kyrollos/LearnedMiniscope3D/RandoscopePSFS/SVD_2_5um_PSF_5um_1_ds4_dsz1_comps_green_SubAvg.mat') 
    weights=scipy.io.loadmat('/home/kyrollos/LearnedMiniscope3D/RandoscopePSFS/SVD_2_5um_PSF_5um_1_ds4_dsz1_weights_green_SubAvg.mat')

    depth_plane=0 #NOTE Z here is 1 less than matlab file as python zero index. So this is z31 in matlab

    h=h['array_out']
    weights=weights['array_out']
    # make sure its (x,y,z,r)
    h=np.swapaxes(h,2,3)
    weights=np.swapaxes(weights,2,3)

    h=h[:,:,depth_plane,:]
    weights=weights[:,:,depth_plane,:]

    # Normalize weights to have maximum sum through rank of 1
    weights_norm = np.max(np.sum(weights[weights.shape[0]//2-1,weights.shape[1]//2-1,:],0))
    weights = weights/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=h/np.linalg.norm(np.ravel(h))

    # padded values for 2D

    ccL = h.shape[1]//2
    ccU = 3*h.shape[1]//2
    rcL = h.shape[0]//2
    rcU = 3*h.shape[0]//2

    H=np.ndarray((h.shape[0]*2,h.shape[1]*2,h.shape[2]), dtype=complex)
    Hconj=np.ndarray((h.shape[0]*2,h.shape[1]*2,h.shape[2]),dtype=complex)
    for i in range (h.shape[2]):
        H[:,:,i]=(np.fft.fft2(pad2d(h[:,:,i])))
        Hconj[:,:,i]=(np.conj(H[:,:,i]))
    return H,weights,[rcL,rcU,ccL,ccU]

# load in forward model weights
def load_weights_3d(path_psfs, path_weights):
    h=scipy.io.loadmat(path_psfs) 
    weights=scipy.io.loadmat(path_weights)

    h=h['array_out']
    weights=weights['array_out']
    #make the shape, xyzr
    h=np.swapaxes(h,2,3)
    weights=np.swapaxes(weights,2,3)
    h=h[:,:,::2,:]
    h=h[:,:,0:32,:]
    weights=weights[:,:,::2,:]
    weights=weights[:,:,0:32,:]

    # Normalize weights to have maximum sum through rank of 1
    weights_norm = np.sum(weights[weights.shape[0]//2-1,weights.shape[1]//2-1,:],0).max()
    weights = weights/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=h/(np.linalg.norm(h.ravel()))

    ccL = h.shape[1]//2
    ccU = 3*h.shape[1]//2
    rcL = h.shape[0]//2
    rcU = 3*h.shape[0]//2

    crop_indices = [rcL,rcU,ccL,ccU]

    H=np.fft.fft2(pad4d(h), axes = (0,1))
    Hconj=np.conj(H)

    return H,weights,crop_indices
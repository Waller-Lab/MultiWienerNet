import numpy as np

def soft_py(x, tau):
    threshed = np.maximum(np.abs(x)-tau, 0)
    threshed = threshed*np.sign(x)
    return threshed

def ht3(x, ax, shift, thresh):
    C = 1./np.sqrt(2.)
    
    if shift == True:
        x = np.roll(x, -1, axis = ax)
    if ax == 0:
        w1 = C*(x[1::2,...] + x[0::2, ...])
        w2 = soft_py(C*(x[1::2,...] - x[0::2, ...]), thresh)
    elif ax == 1:
        w1 = C*(x[:, 1::2] + x[:, 0::2])
        w2 = soft_py(C*(x[:,1::2] - x[:,0::2]), thresh)
    elif ax == 2:
        w1 = C*(x[:,:,1::2] + x[:,:, 0::2])
        w2 = soft_py(C*(x[:,:,1::2] - x[:,:,0::2]), thresh)
    return w1, w2

def iht3(w1, w2, ax, shift, shape):
    
    C = 1./np.sqrt(2.)
    y = np.zeros(shape)

    x1 = C*(w1 - w2); x2 = C*(w1 + w2); 
    if ax == 0:
        y[0::2, ...] = x1
        y[1::2, ...] = x2
     
    if ax == 1:
        y[:, 0::2] = x1
        y[:, 1::2] = x2
    if ax == 2:
        y[:, :, 0::2] = x1
        y[:, :, 1::2] = x2
        
    
    if shift == True:
        y = np.roll(y, 1, axis = ax)
    return y


def iht3_py2(w1, w2, ax, shift, shape):
    
    C = 1./np.sqrt(2.)
    y = np.zeros(shape)

    x1 = C*(w1 - w2); x2 = C*(w1 + w2); 
        
    ind = ax + 2;
    y = np.reshape(np.concatenate([np.expand_dims(x1, ind), np.expand_dims(x2, ind)], axis = ind), shape)
    
    
    if shift == True:
        y = np.roll(y, 1, axis = ax+1)
    return y

def tv3dApproxHaar(x, tau, alpha):
    D = 3
    fact = np.sqrt(2)*2

    thresh = D*tau*fact
    
    sqeezed = False
    if x.shape[-1] == 1:
        x = x[...,0]
        sqeezed = True

    y = np.zeros_like(x)
    for ax in range(0,len(x.shape)):
        if ax ==2:
            t_scale = alpha
        else:
            t_scale = 1;

        w0, w1 = ht3(x, ax, False, thresh*t_scale)
        w2, w3 = ht3(x, ax, True, thresh*t_scale)
        
        t1 = iht3(w0, w1, ax, False, x.shape)
        t2 = iht3(w2, w3, ax, True, x.shape)
        y = y + t1 + t2
        
    y = y/(2*D)
    if sqeezed == True:
        y = y[..., np.newaxis]
    return y




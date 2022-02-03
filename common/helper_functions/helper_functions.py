import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from IPython.core.display import display, HTML
from ipywidgets import interact, widgets, fixed

import sys
sys.path.append('helper_functions/')
   
    
def plotf2(r, img, ttl, sz):
    plt.title(ttl+' {}'.format(r))
    plt.imshow(img[:,:,r], vmin = np.min(img), vmax = np.max(img));
    plt.axis('off');
    fig = plt.gcf()
    fig.set_size_inches(sz)
    plt.show();
    return 

def plt3D(img, title = '', size = (5,5)):
    interact(plotf2, 
             r=widgets.IntSlider(min=0,max=np.shape(img)[-1]-1,step=1,value=1), 
             img = fixed(img), 
             continuous_update= False, 
             ttl = fixed(title), 
             sz = fixed(size));
    
    
    
def plotf22(r, img, ttl, sz):
    plt.title(ttl+' {}'.format(r))
    plt.imshow(img[:,:,r], vmin = np.min(img[:,:,r]), vmax = np.max(img[:,:,r]));
    plt.axis('off');
    fig = plt.gcf()
    fig.set_size_inches(sz)
    plt.show();
    return 

def plt3D2(img, title = '', size = (5,5)):
    interact(plotf22, 
             r=widgets.IntSlider(min=0,max=np.shape(img)[-1]-1,step=1,value=1), 
             img = fixed(img), 
             continuous_update= False, 
             ttl = fixed(title), 
             sz = fixed(size));
    
def crop(x):
    DIMS0 = x.shape[0]//2  # Image Dimensions
    DIMS1 = x.shape[1]//2  # Image Dimensions

    PAD_SIZE0 = int((DIMS0)//2)                           # Pad size
    PAD_SIZE1 = int((DIMS1)//2)                           # Pad size

    C01 = PAD_SIZE0; C02 = PAD_SIZE0 + DIMS0              # Crop indices 
    C11 = PAD_SIZE1; C12 = PAD_SIZE1 + DIMS1              # Crop indices 
    return x[C01:C02, C11:C12,:]

def pre_plot(x):
    x = np.fliplr(np.flipud(x))
    x = x/np.max(x)
    x = np.clip(x, 0,1)
    return x
    
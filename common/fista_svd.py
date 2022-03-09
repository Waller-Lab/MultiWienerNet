import sys
global device
device= sys.argv[1]
sys.path.append('helper_functions/')
import forward_model as fm

if device == 'GPU':
    import cupy as np
    import tv_approx_haar_cp as tv
    
    print('device = ', device, ', using GPU and cupy')
else:
    import numpy as np
    import tv_approx_haar_np as tv
    print('device = ', device, ', using CPU and numpy')

import helper_functions.helper_functions as fc
import numpy as numpy
import matplotlib.pyplot as plt



class fista_svd():
    def __init__(self, H, weights, crop_indices, obj_type = '2D_svd'):
        
        ## Initialize constants 
        self.DIMS0 = weights.shape[0]  # Image Dimensions
        self.DIMS1 = weights.shape[1]  # Image Dimensions
        
        self.py = int((self.DIMS0)//2)    # Pad size
        self.px = int((self.DIMS1)//2)    # Pad size
        
        # FFT of point spread function 
        self.H = H#np.expand_dims(np.fft.fft2((np.fft.ifftshift(self.pad(h), axes = (0,1))), axes = (0,1)), -1)
        self.Hconj = np.conj(self.H)  
        
        self.weights = weights
        self.crop_indices = crop_indices
        #self.mask = mask
       
        if obj_type == '2D_svd':
            self.Apow = fm.A_2d_svd_power
            self.A = fm.A_2d_svd_crop
            self.Aadj = fm.A_2d_adj_svd
            self.pad = fm.pad2d
            self.im_dims = [weights.shape[0]*2, weights.shape[1]*2]
        elif obj_type == '2D':
            self.Apow = fm.A_2d_power
            self.A = fm.A_2d_crop
            self.Aadj = fm.A_2d_adj
            self.pad = fm.pad2d
            self.im_dims = [weights.shape[0]*2, weights.shape[1]*2]
        elif obj_type == '3D':
            self.Apow = fm.A_3d_power
            self.A = fm.A_3d_crop
            self.Aadj = fm.A_3d_adj_fista
            self.pad = fm.pad2d
            self.im_dims = [weights.shape[0]*2, weights.shape[1]*2, H.shape[2]]
        elif obj_type == '3D_svd':
            self.Apow = fm.A_3d_svd_power
            self.A = fm.A_3d_svd_crop
            self.Aadj = fm.A_3d_adj_svd
            self.pad = fm.pad2d
            self.im_dims = [weights.shape[0]*2, weights.shape[1]*2, H.shape[2]]
        else:
            print('invalid object type')
    
        # Calculate the eigenvalue to set the step size 
        maxeig =  self.power_iteration(self.Apow, (self.im_dims[0:2]), 10)
        self.L =  maxeig* 2 #45
        
        
        self.prox_method = 'tv'  # options: 'non-neg', 'tv', 'native'
        
        # Define soft-thresholding constants
        self.tau = .5                  # Native sparsity tuning parameter
        self.tv_lambda = 0.00005       # TV tuning parameter
        self.tv_lambdaw = 0.00005      # TV tuning parameter for wavelength 
        self.lowrank_lambda = 0.00005  # Low rank tuning parameter
       
        
        # Number of iterations of FISTA
        self.iters = 500
        
        self.show_recon_progress = True # Display the intermediate results
        self.print_every = 20           # Sets how often to print the image
        
        self.l_data = []
        self.l_tv = []
        self.obj_type = obj_type
        
    # Power iteration to calculate eigenvalue 
    def power_iteration(self, A, sample_vect_shape, num_iters):
        bk = np.random.randn(*sample_vect_shape)
        for i in range(0, num_iters):
            bk1 = A(bk, self.H, self.weights,fm.pad2d)
            bk1_norm = np.linalg.norm(bk1)

            bk = bk1/bk1_norm
        Mx = A(bk,self.H, self.weights,fm.pad2d)
        xx = np.transpose(np.dot(bk.ravel(), bk.ravel()))
        eig_b = np.transpose(bk.ravel()).dot(Mx.ravel())/xx

        return eig_b
            
    # Helper functions for forward model 
    def crop(self,x):
        return x[self.py:-self.py, self.px:-self.px]
    
    def pad(self,x):
        if len(x.shape) == 2: 
            out = np.pad(x, ([self.py, self.py], [self.px,self.px]), mode = 'constant')
        elif len(x.shape) == 3:
            out = np.pad(x, ([self.py, self.py], [self.px,self.px], [0, 0]), mode = 'constant')
        return out

    def soft_thresh(self, x, tau):
        out = np.maximum(np.abs(x)- tau, 0)
        out = out*np.sign(x)
        return out 
    
    def prox(self,x):
        if self.prox_method == 'tv':
            x = 0.5*(np.maximum(x,0) + tv.tv3dApproxHaar(x, self.tv_lambda/self.L, self.tv_lambdaw))
        if self.prox_method == 'native':
            x = 0.5*(np.maximum(x,0) + self.soft_thresh(x, self.tau))
        if self.prox_method == 'non-neg':
            x = np.maximum(x,0) 
        return x
        
    def tv(self, x):
        d = np.zeros_like(x)
        d[0:-1,:] = (x[0:-1,:] - x[1:, :])**2
        d[:,0:-1] = d[:,0:-1] + (x[:,0:-1] - x[:,1:])**2
        return np.sum(np.sqrt(d))
        
    def loss(self,x,err):
        if self.prox_method == 'tv':
            self.l_data.append(np.linalg.norm(err)**2)
            self.l_tv.append(2*self.tv_lambda/self.L * self.tv(x))
            
            l = np.linalg.norm(err)**2 + 2*self.tv_lambda/self.L * self.tv(x)
        if self.prox_method == 'native':
            l = np.linalg.norm(err)**2 + 2*self.tv_lambda/self.L * np.linalg.norm(x.ravel(), 1)
        if self.prox_method == 'non-neg':
            l = np.linalg.norm(err)**2
        return l
        
    # Main FISTA update 
    def fista_update(self, vk, tk, xk, inputs):

        error = self.A(vk, self.H, self.weights, self.pad, self.crop_indices) - inputs
        grads = self.Aadj(self.Hconj,self.weights,error,self.pad)
    
        xup = self.prox(vk - 1/self.L * grads)
        tup = 1 + np.sqrt(1 + 4*tk**2)/2
        vup = xup + (tk-1)/tup * (xup-xk)
            
        return vup, tup, xup, self.loss(xup, error)


    # Run FISTA 
    def run(self, inputs):   

        # Initialize variables to zero 
        xk = np.zeros((self.im_dims))
        vk = np.zeros((self.im_dims))
        tk = 1.0
        
        llist = []

        # Start FISTA loop 
        for i in range(0,self.iters):
            vk, tk, xk, l = self.fista_update(vk, tk, xk, inputs)
        
            if device == 'GPU':
                l =l.get()
                
            llist.append(l)
        
            # Print out the intermediate results and the loss 
            if self.show_recon_progress== True and i%self.print_every == 0:
                print('iteration: ', i, ' loss: ', l)
                if device == 'GPU':
                    out_img = np.asnumpy(self.crop(xk))
                else:
                    out_img = self.crop(xk)
                
                if len(out_img.shape)==3:
                    fc_img = numpy.max(numpy.real(out_img),-1)
                else:
                    fc_img = out_img
                    
                    
                plt.figure(figsize = (10,3))
                plt.subplot(1,2,1), plt.imshow(fc_img/numpy.max(fc_img)); plt.title('Reconstruction')
                plt.subplot(1,2,2), plt.plot(llist); plt.title('Loss')
                plt.show()
                self.out_img = out_img
        xout = self.crop(xk) 
        return xout, llist
import numpy as np
from scipy.fftpack import dct, idct

import scipy.io
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
from scipy.interpolate import griddata



def  register_psfs(stack,ref_im,dct_on=True):

    [Ny, Nx] = stack[:,:,0].shape;
    vec = lambda x: x.ravel()
    pad2d = lambda x: np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2)),'constant', constant_values=(0))
    fftcorr = lambda x,y:np.fft.ifft2(np.fft.fft2(pad2d(x))*np.conj(np.fft.fft2(np.fft.ifftshift(pad2d(y)))));
    M = stack.shape[2]
    Si = lambda x,si:np.roll(np.roll(x,si[0],axis=0),si[1],axis=1);

    pr = Ny + 1;
    pc = Nx + 1; # Relative centers of all correlations

    yi_reg = 0*stack;   #Registered stack
    pad = lambda x:x;
    crop = lambda x:x;
    pad2d = lambda x:np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2)),'constant', constant_values=(0))
    crop2d = lambda x: x[Ny//2:3*Ny//2,Nx//2:3*Nx//2];


    #     % Normalize the stack first
    stack_norm = np.zeros((1,M));
    stack_dct = stack*1;
    ref_norm = np.linalg.norm(ref_im,'fro');
    for m in range (M):
        stack_norm[0,m] = np.linalg.norm(stack_dct[:,:,m],'fro');
        stack_dct[:,:,m] = stack_dct[:,:,m]/stack_norm[0,m];
        stack[:,:,m] = stack[:,:,m]/ref_norm;

    ref_im = ref_im/ref_norm;


    #     #########
    si={}

    # #     % Do fft registration


    if dct_on:
        print('Removing background\n')
        for n in range (stack_dct.shape[2]):
            im = stack_dct[:,:,n];
            bg_dct = dct(im);
            bg_dct[0:19,0:19] = 0;

            stack_dct[:,:,n] = idct(np.reshape(bg_dct,im.shape));


        print('done\n')
    roi=np.zeros((Ny,Nx))
    print('registering\n')
    good_count = 0;

    for m in range (M):

        corr_im = np.real(fftcorr(stack_dct[:,:,m],ref_im));

        if np.max(corr_im) < .01:
            print('image %i has poor match. Skipping\n',m);
        else:

            [r,c] =np.unravel_index(np.argmax(corr_im),(2*Ny,2*Nx))

            si[good_count] = [-(r-pr),-(c-pc)];


            W = crop2d(Si(np.logical_not(pad2d(np.logical_not(roi))),-np.array(si[good_count])));

            bg_estimate = np.sum(np.sum(W*stack[:,:,m]))/np.maximum(np.count_nonzero(roi),1)*0;
            im_reg = ref_norm*crop(Si(pad(stack[:,:,m]-bg_estimate),si[good_count]));


            yi_reg[:,:,good_count] = im_reg;
            good_count = good_count + 1;


    yi_reg = yi_reg[:,:,0:good_count];


    print('done registering\n')
    
    
    return yi_reg,si

def calc_svd(yi_reg,si,rnk):    
    [Ny, Nx] = yi_reg[:,:,0].shape;
    print('creating matrix\n')
    Mgood = yi_reg.shape[2];
    ymat = np.zeros((Ny*Nx,Mgood));
    ymat=yi_reg.reshape(( yi_reg.shape[0]* yi_reg.shape[1], yi_reg.shape[2]),order='F')

    print('done\n')

    print('starting svd...\n')

    print('check values of ymat')
    [u,s,v] = svds(ymat,rnk);


    comps = np.reshape(u,[Ny, Nx,rnk],order='F');
    vt = v*1
    # s=np.flip(s)
    # vt=np.flipud(vt)
    weights = np.zeros((Mgood,rnk));
    for m  in range (Mgood):
        for r in range(rnk):
            weights[m,r]=s[r]*vt[r,m]


    # si_mat = reshape(cell2mat(si)',[2,Mgood]);
    xq = np.arange(-Nx/2,Nx/2);
    yq = np.arange(-Ny/2,Ny/2);
    [Xq, Yq] = np.meshgrid(xq,yq);

    weights_interp = np.zeros((Ny, Nx,rnk));
    xi=[]
    yi=[]
    si_list=list(si.values())

    for i in range(len(si_list)):
        xi.append(si_list[i][0])
        yi.append(si_list[i][1])

    print('interpolating...\n')
    for r in range(rnk):
    #     interpolant_r = scatteredInterpolant(si_mat(2,:)', si_mat(1,:)', weights(:,r),'natural','nearest');
    #     weights_interp(:,:,r) = rot90(interpolant_r(Xq,Yq),2);
        weights_interp[:,:,r]=griddata((xi,yi),weights[:,r],(Xq,Yq),method='nearest')

    print('done\n\n')

    return np.flip(comps,-1), np.flip(weights_interp,-1)


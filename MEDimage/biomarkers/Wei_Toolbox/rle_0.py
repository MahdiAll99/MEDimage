#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def rle_0(si,NL):
    """Compute rle_0
    RLE: image gray level Run Length matrix for 0degree
    -------------------------------------------------------------------------
    AUTHOR(S): - MEDomicsLab consortium
               - Adapted from MATLAB code of Xunkai Wei <xunkai.wei@gmail.com>
                 Beijing Aeronautical Technology Research Center
    -------------------------------------------------------------------------
    """

    # Assure row number is exactly the gray level
    m,n =np.shape(si)
    oneglrlm = np.zeros((NL,n)).astype('int')
    
    for i in range(1,m+1):
        x=si[i-1,:]
        # run length Encode of each vector
        index = np.append(np.nonzero(x[:-1] != x[1:])[0]+1, n)
        le = np.diff(np.append(0,index)) # run lengths
        val = (x[index-1]).astype('int') # run values
        # compute current numbers (or contribution) for each bin in GLRLM
        # accumulate each contribution
        for x_val, y_le in zip(val, le): oneglrlm[x_val-1,y_le-1]+=1 
   
    return oneglrlm





















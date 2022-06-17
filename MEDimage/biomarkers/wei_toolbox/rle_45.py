#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def rle_45(seq, NL):
    """Compute rle_45.
    
    RLE   image gray level Run Length matrix for 45 and 135
    This file is to handle the zigzag scanned sequence for 45 or 135 degree
    direction. 
    
    Note:
        For 135, just swap the left and the right colum.

    AUTHOR(S): 
        
        - MEDomicsLab consortium
        - Adapted from MATLAB code of Xunkai Wei <xunkai.wei@gmail.com>
            Beijing Aeronautical Technology Research Center

    """

    # Assure row number is exactly the gray level
    # number of seqence
    m = len(seq)

    #n = findmaxnum(seq)
    # number to store the possible max coloums
    n = max(np.size(row) for row in seq.values())
    oneglrlm = np.zeros((NL, n)).astype('int')

    for i in range(1, m+1):
        x = np.array(seq[i-1], ndmin=1)
        # run length Encode of each vector
        index = np.append(np.nonzero(x[:-1] != x[1:])[0]+1, len(x))
        le = np.diff(np.append(0, index))  # run lengths
        val = (x[index-1]).astype('int')  # run values
        # compute current numbers (or contribution) for each bin in GLRLM
        # accumulate each contribution
        for x_val, y_le in zip(val, le):
            oneglrlm[x_val-1, y_le-1] += 1

    return oneglrlm

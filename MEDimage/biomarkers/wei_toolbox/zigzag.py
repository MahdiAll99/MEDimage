#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def zigzag(SI):
    """Compute zigzag (SI)
    
    This function is used to build the corresponding sequences of a given
    scaled gray level image matrix from 45' degree direction.
    The whole process is using zigzag method. It can handle nonsquare 
    image matrix. Trick: all the sequence starts or ends lie on the boundary.
    
    AUTHOR(S): 
        
        - MEDomicsLab consortium
        - Adapted from MATLAB code of Xunkai Wei <xunkai.wei@gmail.com>
            Beijing Aeronautical Technology Research Center

    """
    seq = {}

    # initializing the variables
    c = 0  # initialize colum indicator
    r = 0  # initialize row   indicator

    rmin = 0  # row boundary checker
    cmin = 0  # colum boundary checker

    rmax = np.shape(SI)[0] - 1  # get row numbers
    cmax = np.shape(SI)[1] - 1  # get colum numbers

    i = 0  # counter for current ith element
    j = 0  # indicator for determining sequence interval

    # intialize sequence mark
    sq_up_begin = 0
    sq_down_begin = 0

    # Output contain value and its flag status
    #  the first row contain value
    #  the second row contain its flag
    output = np.zeros((rmax+1)*(cmax+1))

    while (r <= rmax) & (c <= cmax):
        # for current point, judge its zigzag direction up 45, or down 45, or
        # 0,or down 90
        if np.mod(c + r, 2) == 0:  # up 45 direction
            # if we currently walk to the left first colum
            if r == rmin:
                # First, record current point
                output[i] = SI[r, c]
                # if we walk to right last colum
                if c == cmax:
                    # add row number move straight down 90
                    r += 1
                    sq_up_end = i
                    sq_down_begin = i+1
                    seq[j] = output[sq_up_begin:sq_up_end+1]
                    j += 1
                else:
                    # Continue to move to next (1,c+1) point
                    # This next point should be the begin point
                    # of next sequence
                    c += 1
                    sq_up_end = i
                    sq_down_begin = i+1
                    seq[j] = output[sq_up_begin:sq_up_end+1]
                    j += 1
                # add couter
                i += 1
                # if we currently walk to the last column
            elif (c == cmax) & (r < rmax):
                # first record the point
                output[i] = SI[r, c]
                # then move straight down to next row
                r += 1
                sq_up_end = i
                seq[j] = output[sq_up_begin:sq_up_end+1]
                sq_down_begin = i+1
                j += 1
                # add counter
                i += 1
                # all other cases i.e. nonboundary points
            elif (r > rmin) & (c < cmax):
                output[i] = SI[r, c]
                # move to next up 45 point
                r -= 1
                c += 1
                # add counter
                i += 1
                # down 45 direction
        else:
            # if we walk to the last row
            if (r == rmax) & (c <= cmax):
                # firstly record current point
                output[i] = SI[r, c]
                # move right to next point
                c += 1
                sq_down_end = i
                seq[j] = output[sq_down_begin:sq_down_end+1]
                sq_up_begin = i+1
                j += 1
                # add counter
                i += 1
                # if we walk to the first column
            elif c == cmin:
                # first record current point
                output[i] = SI[r, c]
                if r == rmax:
                    c += 1
                    sq_down_end = i
                    seq[j] = output[sq_down_begin:sq_down_end+1]
                    sq_up_begin = i+1
                    j += 1
                else:
                    r += 1
                    # record sequence end
                    sq_down_end = i
                    seq[j] = output[sq_down_begin:sq_down_end+1]
                    sq_up_begin = i+1
                    j += 1
                # add counter
                i += 1
                # all other cases without boundary point
            elif (r < rmax) & (c > cmin):
                output[i] = SI[r, c]
                r += 1
                c -= 1
                # keep down_info
                i += 1

        if (r == rmax) & (c == cmax):  # bottom right element
            output[i] = SI[r, c]
            sq_end = i
            seq[j] = output[sq_end]
            break

    return seq

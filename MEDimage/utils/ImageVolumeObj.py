#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class ImageVolumeObj:
    """Used to organize Imaging data and their corresponding imref3d object. 

    Args:
        data (:obj:`ndarray`, optional): 3D array of imaging data.
        spatialRef (:obj:`imref3d`, optional): The corresponding imref3d object 
            (same functionality of MATLAB imref3d class).

    Attributes:
        data (ndarray): 3D array of imaging data.
        spatialRef (imref3d): The corresponding imref3d object 
            (same functionality of MATLAB imref3d class).

    """

    def __init__(self, data=None, spatialRef=None) -> None:
        self.data = data
        self.spatialRef = spatialRef

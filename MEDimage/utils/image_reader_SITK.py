#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import SimpleITK as sitk
import numpy as np


def image_reader_SITK(path, option=None):
    """Return the image in a numpy array or a dictionary with the header
    of the image.

    Args:
        path (path): path of the file
        option ():
    
    Returns:
        Union[None, Dict]: 
    """
    if option is None or option == 'image':
        # return the image in a numpy array
        return np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(path)))
    elif option == 'header':
        # Return a dictionary with the header of the image.
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        # reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        dic_im_header = {}
        for key in reader.GetMetaDataKeys():
            dic_im_header.update({key: reader.GetMetaData(key)})
        return dic_im_header
    else:
        print("Argument option should be the string 'image' or 'header'")
        return None

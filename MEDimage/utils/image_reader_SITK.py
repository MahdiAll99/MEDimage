#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import SimpleITK as sitk
import numpy as np


def image_reader_SITK(path, option=None):
    """
    Return the image in a numpy array or a dictionary with the header
    of the image.
    -------------------------------------------------------------------------
    AUTHOR(S): MEDomicsLab consortium
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
    a package providing MATLAB programming tools for radiomics analysis.
     --> Copyright (C) MEDomicsLab consortium.

    This package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package.  If not, see <http://www.gnu.org/licenses/>.
    -------------------------------------------------------------------------
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

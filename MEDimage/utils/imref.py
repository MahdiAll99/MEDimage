#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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

from typing import Union

import numpy as np


def intrinsicToWorld(R, xIntrinsic, yIntrinsic, zIntrinsic):
    return R.intrinsicToWorld(xIntrinsic=xIntrinsic, yIntrinsic=yIntrinsic, zIntrinsic=zIntrinsic)


def worldToIntrinsic(R, xWorld, yWorld, zWorld):
    return R.worldToIntrinsic(xWorld=xWorld, yWorld=yWorld, zWorld=zWorld)


def sizesMatch(R, A):
    """Compares whether the two imref3d objects have the same size.

    Args:
        R (imref3d): First imref3d object.
        A (imref3d): Second imref3d object.

    Returns:
        bool: True if R and A have the same size, and false if not.

    """
    return np.all(R.imageSize == A.imageSize)


class imref3d:
    """This class mirrors the functionality of the matlab imref3d class

    An imref3d object stores the relationship between the intrinsic coordinates 
    anchored to the columns,  rows, and planes of a 3-D image and the spatial 
    location of the same column, row, and plane locations in a world coordinate system.

    The image is sampled regularly in the planar world-x, world-y, and world-z coordinates 
    of the coordinate system such that intrinsic-x, -y and -z values align with world-x, -y 
    and -z values, respectively. The resolution in each dimension can be different.
    REF: <https://www.mathworks.com/help/images/ref/imref3d.html>

    Args:
        ImageSize (:obj:`ndarray`, optional): Number of elements in each spatial dimension, 
            specified as a 3-element positive row vector.
        PixelExtentInWorldX (:obj:`float`, optional): Size of a single pixel in the x-dimension 
            measured in the world coordinate system.
        PixelExtentInWorldY (:obj:`float`, optional): Size of a single pixel in the y-dimension 
            measured in the world coordinate system.
        PixelExtentInWorldZ (:obj:`float`, optional): Size of a single pixel in the z-dimension 
            measured in the world coordinate system.
        xWorldLimits (:obj:`ndarray`, optional): Limits of image in world x, specified as a 2-element row vector, 
            [xMin xMax].
        yWorldLimits (:obj:`ndarray`, optional): Limits of image in world y, specified as a 2-element row vector, 
            [yMin yMax].
        zWorldLimits (:obj:`ndarray`, optional): Limits of image in world z, specified as a 2-element row vector, 
            [zMin zMax].
        
    Attributes:
        ImageSize (:obj:`ndarray`): Number of elements in each spatial dimension, 
            specified as a 3-element positive row vector.
        PixelExtentInWorldX (:obj:`float`): Size of a single pixel in the x-dimension 
            measured in the world coordinate system.
        PixelExtentInWorldY (:obj:`float`): Size of a single pixel in the y-dimension 
            measured in the world coordinate system.
        PixelExtentInWorldZ (:obj:`float`): Size of a single pixel in the z-dimension 
            measured in the world coordinate system.
        XIntrinsicLimits (:obj:`ndarray`): Limits of image in intrinsic units in the x-dimension, 
            specified as a 2-element row vector [xMin xMax].
        YIntrinsicLimits (:obj:`ndarray`): Limits of image in intrinsic units in the y-dimension, 
            specified as a 2-element row vector [yMin yMax].
        ZIntrinsicLimits (:obj:`ndarray`): Limits of image in intrinsic units in the z-dimension, 
            specified as a 2-element row vector [zMin zMax].
        ImageExtentInWorldX (:obj:`float`): Span of image in the x-dimension in 
            the world coordinate system.
        ImageExtentInWorldY (:obj:`float`): Span of image in the y-dimension in 
            the world coordinate system.
        ImageExtentInWorldZ (:obj:`float`): Span of image in the z-dimension in 
            the world coordinate system.
        xWorldLimits (:obj:`ndarray`): Limits of image in world x, specified as a 2-element row vector, 
            [xMin xMax].
        yWorldLimits (:obj:`ndarray`): Limits of image in world y, specified as a 2-element row vector, 
            [yMin yMax].
        zWorldLimits (:obj:`ndarray`): Limits of image in world z, specified as a 2-element row vector, 
            [zMin zMax].

    """

    def __init__(self, 
                imageSize=None, 
                pixelExtentInWorldX=1.0,
                pixelExtentInWorldY=1.0, 
                pixelExtentInWorldZ=1.0,
                xWorldLimits=None, 
                yWorldLimits=None, 
                zWorldLimits=None) -> None:

        # Check if imageSize is an ndarray, and cast to ndarray otherwise
        self.ImageSize = self._parse_to_ndarray(x=imageSize, n=3)

        # Size of single voxels along axis in world coordinate system.
        # Equivalent to voxel spacing.
        self.PixelExtentInWorldX = pixelExtentInWorldX
        self.PixelExtentInWorldY = pixelExtentInWorldY
        self.PixelExtentInWorldZ = pixelExtentInWorldZ

        # Limits of the image in intrinsic coordinates
        # AZ: this differs from DICOM, which assumes that the origin lies
        # at the center of the first voxel.
        if imageSize is not None:
            self.XIntrinsicLimits = np.array([-0.5, imageSize[0]-0.5])
            self.YIntrinsicLimits = np.array([-0.5, imageSize[1]-0.5])
            self.ZIntrinsicLimits = np.array([-0.5, imageSize[2]-0.5])
        else:
            self.XIntrinsicLimits = None
            self.YIntrinsicLimits = None
            self.ZIntrinsicLimits = None

        # Size of the image in world coordinates
        if imageSize is not None:
            self.ImageExtentInWorldX = imageSize[0] * pixelExtentInWorldX
            self.ImageExtentInWorldY = imageSize[1] * pixelExtentInWorldY
            self.ImageExtentInWorldZ = imageSize[2] * pixelExtentInWorldZ
        else:
            self.ImageExtentInWorldX = None
            self.ImageExtentInWorldY = None
            self.ImageExtentInWorldZ = None

        # Limits of the image in the world coordinates
        self.XWorldLimits = self._parse_to_ndarray(x=xWorldLimits, n=2)
        self.YWorldLimits = self._parse_to_ndarray(x=yWorldLimits, n=2)
        self.ZWorldLimits = self._parse_to_ndarray(x=zWorldLimits, n=2)

        if xWorldLimits is None and imageSize is not None:
            self.XWorldLimits = np.array([0.0, self.ImageExtentInWorldX])
        if yWorldLimits is None and imageSize is not None:
            self.YWorldLimits = np.array([0.0, self.ImageExtentInWorldY])
        if zWorldLimits is None and imageSize is not None:
            self.ZWorldLimits = np.array([0.0, self.ImageExtentInWorldZ])

    def _parse_to_ndarray(self, x, n=None) -> np.ndarray:
        """Internal function to cast input to a numpy array.

        Args:
            x (iterable): Object that supports __iter__.
            n (:obj:`int`, optional): expected length.

        Returns:
            array: iterable input as a numpy array.
        
        """
        if x is not None:
            # Cast to ndarray
            if not isinstance(x, np.ndarray):
                x = np.array(x)

            # Check length
            if n is not None:
                if not len(x) == n:
                    raise ValueError(
                        "Length of array does not meet the expected length.", len(x), n)

        return x

    def intrinsicToWorld(self, 
                        xIntrinsic, 
                        yIntrinsic, 
                        zIntrinsic
                        ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Convert from intrinsic to world coordinates.

        Args:
            xIntrinsic (ndarray): Coordinates along the x-dimension in 
                the intrinsic coordinate system.
            yIntrinsic (ndarray): Coordinates along the y-dimension in 
                the intrinsic coordinate system.
            zIntrinsic (ndarray): Coordinates along the z-dimension in 
                the intrinsic coordinate system.

        Returns:
            array: [xWorld, yWorld, zWorld] in world coordinate system.
        
        """
        xWorld = (self.XWorldLimits[0] + 0.5*self.PixelExtentInWorldX) + \
            xIntrinsic * self.PixelExtentInWorldX
        yWorld = (self.YWorldLimits[0] + 0.5*self.PixelExtentInWorldY) + \
            yIntrinsic * self.PixelExtentInWorldY
        zWorld = (self.ZWorldLimits[0] + 0.5*self.PixelExtentInWorldZ) + \
            zIntrinsic * self.PixelExtentInWorldZ

        return xWorld, yWorld, zWorld

    def worldToIntrinsic(self, 
                        xWorld, 
                        yWorld, 
                        zWorld
                        )-> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Converts from world coordinates to intrinsic coordinates.

        Args:
            xWorld (ndarray): Coordinates along the x-dimension in 
                the world coordinate system.
            yWorld (ndarray): Coordinates along the y-dimension in 
                the world coordinate system.
            zWorld (ndarray): Coordinates along the z-dimension in 
                the world coordinate system.

        Returns:
            array: [xIntrinsic,yIntrinsic,zIntrinsic] in intrinsic 
                coordinate system.

        """

        xIntrinsic = (
            xWorld - (self.XWorldLimits[0] + 0.5*self.PixelExtentInWorldX)) / self.PixelExtentInWorldX
        yIntrinsic = (
            yWorld - (self.YWorldLimits[0] + 0.5*self.PixelExtentInWorldY)) / self.PixelExtentInWorldY
        zIntrinsic = (
            zWorld - (self.ZWorldLimits[0] + 0.5*self.PixelExtentInWorldZ)) / self.PixelExtentInWorldZ

        return xIntrinsic, yIntrinsic, zIntrinsic

    def contains_point(self, xWorld, yWorld, zWorld) -> np.ndarray:
        """Determines which points defined by xWorld, yWorld and zWorld.

        Args:
            xWorld (ndarray): Coordinates along the x-dimension in 
                the world coordinate system.
            yWorld (ndarray): Coordinates along the y-dimension in 
                the world coordinate system.
            zWorld (ndarray): Coordinates along the z-dimension in 
                the world coordinate system.

        Returns:
            array: boolean array for coordinate sets that are within 
                the bounds of the image.

        """
        xInside = np.logical_and(
            xWorld >= self.XWorldLimits[0], xWorld <= self.XWorldLimits[1])
        yInside = np.logical_and(
            yWorld >= self.YWorldLimits[0], yWorld <= self.YWorldLimits[1])
        zInside = np.logical_and(
            zWorld >= self.ZWorldLimits[0], zWorld <= self.ZWorldLimits[1])

        return xInside + yInside + zInside == 3

    def WorldLimits(self, axis=None, newValue=None) -> Union[np.ndarray, None]:
        """Sets the WorldLimits to the new value for the given axis.
        If the newValue is None, the method returns the attribute value.

        Args:
            axis (:obj:`str`, optional): Specify the dimension,
                MUST BE 'X', 'Y' or 'Z'.
            newValue (:obj:`iterable`, optional): New value for the
                WorldLimits attribute.

        Returns:
            array: Limits of image in world along the axis-dimension.

        """
        if newValue is None:
            # Get value
            if axis == "X":
                return self.XWorldLimits
            elif axis == "Y":
                return self.YWorldLimits
            elif axis == "Z":
                return self.ZWorldLimits
        else:
            # Set value
            if axis == "X":
                self.XWorldLimits = self._parse_to_ndarray(x=newValue, n=2)
            elif axis == "Y":
                self.YWorldLimits = self._parse_to_ndarray(x=newValue, n=2)
            elif axis == "Z":
                self.ZWorldLimits = self._parse_to_ndarray(x=newValue, n=2)

    def PixelExtentInWorld(self, axis=None) -> Union[float, None]:
        """Returns the PixelExtentInWorld attribute value for the given axis.

        Args:
            axis (:obj:`str`, optional): Specify the dimension,
                MUST BE 'X', 'Y' or 'Z'.

        Returns:
            float: Size of a single pixel in the axis-dimension measured 
                in the world coordinate system.

        """
        if axis == "X":
            return self.PixelExtentInWorldX
        elif axis == "Y":
            return self.PixelExtentInWorldY
        elif axis == "Z":
            return self.PixelExtentInWorldZ

    def IntrinsicLimits(self, axis=None) -> Union[np.ndarray, None]:
        """Returns the IntrinsicLimits attribute value for the given axis.

        Args:
            axis (:obj:`str`, optional): Specify the dimension,
                MUST BE 'X', 'Y' or 'Z'.

        Returns:
            array: Limits of image in intrinsic units in the axis-dimension, 
                specified as a 2-element row vector [xMin xMax].

        """
        if axis == "X":
            return self.XIntrinsicLimits
        elif axis == "Y":
            return self.YIntrinsicLimits
        elif axis == "Z":
            return self.ZIntrinsicLimits

    def ImageExtentInWorld(self, axis=None) -> Union[float, None]:
        """Returns the ImageExtentInWorld attribute value for the given axis.

        Args:
            axis (:obj:`str`, optional): Specify the dimension,
                MUST BE 'X', 'Y' or 'Z'.

        Returns:
            array: Span of image in the axis-dimension in the world coordinate 
                system.

        """
        if axis == "X":
            return self.ImageExtentInWorldX
        elif axis == "Y":
            return self.ImageExtentInWorldY
        elif axis == "Z":
            return self.ImageExtentInWorldZ

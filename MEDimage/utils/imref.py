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

import numpy as np


def intrinsicToWorld(R, xIntrinsic, yIntrinsic, zIntrinsic):
    return R.intrinsicToWorld(xIntrinsic=xIntrinsic, yIntrinsic=yIntrinsic, zIntrinsic=zIntrinsic)


def worldToIntrinsic(R, xWorld, yWorld, zWorld):
    return R.worldToIntrinsic(xWorld=xWorld, yWorld=yWorld, zWorld=zWorld)


def sizesMatch(R, A):
    """
    Compares whether R and A have the same size
    :param R: an imref3d object
    :param A: another imref3d object
    :return: True if R and A have the same size, and false if not
    """
    return np.all(R.imageSize == A.imageSize)


class imref3d:
    # Mirrors the functionality of the matlab imref3d class
    def __init__(self, imageSize=None, pixelExtentInWorldX=1.0,
                 pixelExtentInWorldY=1.0, pixelExtentInWorldZ=1.0,
                 xWorldLimits=None, yWorldLimits=None, zWorldLimits=None):

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

    def _parse_to_ndarray(self, x, n=None):
        """
        Internal function to cast input to a numpy array
        :param x: input iterable
        :param n: expected length
        :return: iterable as a numpy array
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

    def intrinsicToWorld(self, xIntrinsic, yIntrinsic, zIntrinsic):
        """
        Converts from intrinsic coordinates to world coordinates
        :param xIntrinsic: x intrinsic voxel coordinate
        :param yIntrinsic: y intrinsic voxel coordinate
        :param zIntrinsic: z intrinsic voxel coordinate
        :return: x, y, and z in world coordinates
        """

        xWorld = (self.XWorldLimits[0] + 0.5*self.PixelExtentInWorldX) + \
            xIntrinsic * self.PixelExtentInWorldX
        yWorld = (self.YWorldLimits[0] + 0.5*self.PixelExtentInWorldY) + \
            yIntrinsic * self.PixelExtentInWorldY
        zWorld = (self.ZWorldLimits[0] + 0.5*self.PixelExtentInWorldZ) + \
            zIntrinsic * self.PixelExtentInWorldZ

        return xWorld, yWorld, zWorld

    def worldToIntrinsic(self, xWorld, yWorld, zWorld):
        """
        Converts from world coordinates to intrinsic coordinates
        :param xWorld: x world coordinate
        :param yWorld: y world coordinate
        :param zWorld: z world coordinate
        :return: x, y, and z in intrinsic voxel coordinates
        """

        xIntrinsic = (
            xWorld - (self.XWorldLimits[0] + 0.5*self.PixelExtentInWorldX)) / self.PixelExtentInWorldX
        yIntrinsic = (
            yWorld - (self.YWorldLimits[0] + 0.5*self.PixelExtentInWorldY)) / self.PixelExtentInWorldY
        zIntrinsic = (
            zWorld - (self.ZWorldLimits[0] + 0.5*self.PixelExtentInWorldZ)) / self.PixelExtentInWorldZ

        return xIntrinsic, yIntrinsic, zIntrinsic

    def contains_point(self, xWorld, yWorld, zWorld):
        """
        Determines which points defined by xWorld, yWorld and zWorld
        coordinates are inside the image
        :param xWorld: x world coordinate
        :param yWorld: y world coordinate
        :param zWorld: z world coordinate
        :return: boolean array for coordinate sets that are within the image
        """

        xInside = np.logical_and(
            xWorld >= self.XWorldLimits[0], xWorld <= self.XWorldLimits[1])
        yInside = np.logical_and(
            yWorld >= self.YWorldLimits[0], yWorld <= self.YWorldLimits[1])
        zInside = np.logical_and(
            zWorld >= self.ZWorldLimits[0], zWorld <= self.ZWorldLimits[1])

        return xInside + yInside + zInside == 3

    def WorldLimits(self, axis=None, newValue=None):
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

    def PixelExtentInWorld(self, axis=None):
        if axis == "X":
            return self.PixelExtentInWorldX
        elif axis == "Y":
            return self.PixelExtentInWorldY
        elif axis == "Z":
            return self.PixelExtentInWorldZ

    def IntrinsicLimits(self, axis=None):
        if axis == "X":
            return self.XIntrinsicLimits
        elif axis == "Y":
            return self.YIntrinsicLimits
        elif axis == "Z":
            return self.ZIntrinsicLimits

    def ImageExtentInWorld(self, axis=None):
        if axis == "X":
            return self.ImageExtentInWorldX
        elif axis == "Y":
            return self.ImageExtentInWorldY
        elif axis == "Z":
            return self.ImageExtentInWorldZ

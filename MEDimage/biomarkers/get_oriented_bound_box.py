#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np
import pandas as pd


def rot_matrix(theta: float,
               dim: int=2,
               rot_axis: int=-1) -> np.ndarray:
    """Creates a 2d or 3d rotation matrix

    Args:
        theta (float): angle in radian
        dim (int, optional): dimension size. Defaults to 2.
        rot_axis (int, optional): rotation axis value. Defaults to -1.

    Returns:
        ndarray: rotation matrix
    """

    if dim == 2:
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    elif dim == 3:
        if rot_axis == 0:
            rot_mat = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(theta), -np.sin(theta)],
                                [0.0, np.sin(theta), np.cos(theta)]])
        elif rot_axis == 1:
            rot_mat = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(theta), 0.0, np.cos(theta)]])
        elif rot_axis == 2:
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 1.0]])
        else:
            rot_mat = None
    else:
        rot_mat = None

    return rot_mat


def sig_proc_segmentise(x: List) -> List:
    """Produces a list of segments from input x with values (0,1)

    Args:
        x (List): list of values

    Returns:
        List: list of segments from input x with values (0,1)
    """

    # Create a difference vector
    y = np.diff(x)

    # Find start and end indices of sections with value 1
    ind_1_start = np.array(np.where(y == 1)).flatten()
    if np.shape(ind_1_start)[0] > 0:
        ind_1_start += 1
    ind_1_end = np.array(np.where(y == -1)).flatten()

    # Check for boundary effects
    if x[0] == 1:
        ind_1_start = np.insert(ind_1_start, 0, 0)
    if x[-1] == 1:
        ind_1_end = np.append(ind_1_end, np.shape(x)[0]-1)

    # Generate segment df for segments with value 1
    if np.shape(ind_1_start)[0] == 0:
        df_one = pd.DataFrame({"i":   [],
                               "j":   [],
                               "val": []})
    else:
        df_one = pd.DataFrame({"i":   ind_1_start,
                               "j":   ind_1_end,
                               "val": np.ones(np.shape(ind_1_start)[0])})

    # Find start and end indices for section with value 0
    if np.shape(ind_1_start)[0] == 0:
        ind_0_start = np.array([0])
        ind_0_end = np.array([np.shape(x)[0]-1])
    else:
        ind_0_end = ind_1_start - 1
        ind_0_start = ind_1_end + 1

        # Check for boundary effect
        if x[0] == 0:
            ind_0_start = np.insert(ind_0_start, 0, 0)
        if x[-1] == 0:
            ind_0_end = np.append(ind_0_end, np.shape(x)[0]-1)

        # Check for out-of-range boundary effects
        if ind_0_end[0] < 0:
            ind_0_end = np.delete(ind_0_end, 0)
        if ind_0_start[-1] >= np.shape(x)[0]:
            ind_0_start = np.delete(ind_0_start, -1)

    # Generate segment df for segments with value 0
    if np.shape(ind_0_start)[0] == 0:
        df_zero = pd.DataFrame({"i":   [],
                                "j":   [],
                                "val": []})
    else:
        df_zero = pd.DataFrame({"i":    ind_0_start,
                                "j":    ind_0_end,
                                "val":  np.zeros(np.shape(ind_0_start)[0])})

    df_segm = df_one.append(df_zero).sort_values(by="i").reset_index(drop=True)

    return df_segm


def sig_proc_find_peaks(x: float,
                        ddir: str="pos") -> pd.DataFrame:
    """Determines peak positions in array of values

    Args:
        x (float): value
        ddir (str, optional): positive or negative value. Defaults to "pos".

    Returns:
        pd.DataFrame: peak positions in array of values
    """

    # Invert when looking for local minima
    if ddir == "neg":
        x = -x

    # Generate segments where slope is negative

    df_segm = sig_proc_segmentise(x=(np.diff(x) < 0.0)*1)

    # Start of slope coincides with position of peak (due to index shift induced by np.diff)
    ind_peak = df_segm.loc[df_segm.val == 1, "i"].values

    # Check right boundary
    if x[-1] > x[-2]:
        ind_peak = np.append(ind_peak, np.shape(x)[0]-1)

    # Construct dataframe with index and corresponding value
    if np.shape(ind_peak)[0] == 0:
        df_peak = pd.DataFrame({"ind": [],
                                "val": []})
    else:
        if ddir == "pos":
            df_peak = pd.DataFrame({"ind": ind_peak,
                                    "val": x[ind_peak]})
        if ddir == "neg":
            df_peak = pd.DataFrame({"ind":  ind_peak,
                                    "val": -x[ind_peak]})
    return df_peak

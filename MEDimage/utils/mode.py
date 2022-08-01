#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple, Union

import numpy as np


def mode(x: np.ndarray,
        return_counts=False) -> Union[Tuple[np.ndarray, np.ndarray],
                                      np.ndarray]:
    """Implementation of mode that also returns counts, unlike the standard statistics.mode.

    Args:
        x (ndarray): n-dimensional array of which to find mode.
        return_counts (bool): If True, also return the number of times each unique item appears in ``x``.

    Returns:
        2-element tuple containing
        
        - ndarray: Array of the modal (most common) value in the given array.
        - ndarray: Array of the counts if ``return_counts`` is True.
    """

    unique_values, counts = np.unique(x, return_counts=True)

    if return_counts:
        return unique_values[np.argmax(counts)], np.max(counts)

    return unique_values[np.argmax(counts)]

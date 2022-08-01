#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Union

import numpy as np

from ..utils.strfind import strfind


def parse_contour_string(contour_string) -> Union[Tuple[float, List[str]],
                                                  Tuple[int, List[str]],
                                                  Tuple[List[int], List[str]]]:
    """Finds the delimeters (:math:`'+'` and :math:`'-'`) and the contour indexe(s) from the given string.

    Args:
        contour_string (str, float or int): Index or string of indexes with
        delimeters. For example: :math:`'3'` or :math:`'1-3+2'`.

    Returns:
        float, int: If ``contour_string`` is a an int or float we return ``contour_string``.
        List[str]: List of the delimeters.
        List[int]: List of the contour indexes.

    Example:
        >>> ``contour_string`` = '1-3+2'
        >>> :function: parse_contour_string(contour_string)
        [1, 2, 3], ['+', '-']
        >>> ``contour_string`` = 1
        >>> :function: parse_contour_string(contour_string)
        1, []
    """

    if isinstance(contour_string, (int, float)):
        return contour_string, []

    ind_plus = strfind(string=contour_string, pattern='\+')
    ind_minus = strfind(string=contour_string, pattern='\-')
    ind_operations = np.sort(np.hstack((ind_plus, ind_minus))).astype(int)

    # Parsing operations and contour numbers
    # AZ: I assume that contour_number is an integer
    if ind_operations.size == 0:
        operations = []
        contour_number = [int(contour_string)]
    else:
        n_op = len(ind_operations)
        operations = [contour_string[ind_operations[i]] for i in np.arange(n_op)]

        contour_number = np.zeros(n_op + 1, dtype=int)
        contour_number[0] = int(contour_string[0:ind_operations[0]])
        for c in np.arange(start=1, stop=n_op):
            contour_number[c] = int(contour_string[(ind_operations[c-1]+1) : ind_operations[c]])

        contour_number[-1] = int(contour_string[(ind_operations[-1]+1):])
        contour_number.tolist()

    return contour_number, operations

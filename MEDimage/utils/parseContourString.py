#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Union

import numpy as np

from ..utils.strfind import strfind


def parseContourString(contourString) -> Union[float, int, List[str]]:
    """Finds the delimeters('+' and '-') and the contour indexe(s)
    from the given string.

    Args:
        contourString (str | float | int): Index or string of indexes with
        delimeters. FOR EXAMPLE '3' or '1-3+2'.

    Returns:
        Union[float, int, List[str]]: If `contourString` is a string, the return 
            is a List of strings. If `contourString` is a an int or float we return 
            `contourString`.

    Example:
        >>>contourString = '1-3+2'
        >>>parseContourString(contourString)
        [1, 2, 3], ['+', '-']

        >>>contourString = 1
        >>>parseContourString(contourString)
        1, []
    """

    if isinstance(contourString, (int, float)):
        return contourString, []

    indPlus = strfind(string=contourString, pattern='\+')
    indMinus = strfind(string=contourString, pattern='\-')
    indOperations = np.sort(np.hstack((indPlus, indMinus))).astype(int)

    # Parsing operations and contour numbers
    # AZ: I assume that contourNumber is an integer
    if indOperations.size == 0:
        operations = []
        contourNumber = [int(contourString)]
    else:
        nOp = len(indOperations)
        operations = [contourString[indOperations[i]] for i in np.arange(nOp)]

        contourNumber = np.zeros(nOp + 1, dtype=int)
        contourNumber[0] = int(contourString[0:indOperations[0]])
        for c in np.arange(start=1, stop=nOp):
            contourNumber[c] = int(contourString[(indOperations[c-1]+1) : indOperations[c]])

        contourNumber[-1] = int(contourString[(indOperations[-1]+1):])
        contourNumber.tolist()

    return contourNumber, operations

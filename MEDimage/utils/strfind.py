#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from re import finditer
from typing import List


def strfind(pattern: str,
            string: str) -> List[int]:
    """Finds indices of ``pattern`` in ``string``. Based on regex.

    Note:
        Be careful with + and - symbols. Use :math:`\+` and :math:`\-` instead.

    Args:
        pattern (str): Substring to be searched in the ``string``.
        string (str): String used to find matches.

    Returns:
        List[int]: List of indexes of every occurence of ``pattern`` in the passed ``string``.

    Raises:
        ValueError: If the ``pattern`` does not use backslash with special regex symbols
    """

    if pattern in ('+', '-'):
        raise ValueError(
            "Please use a backslash with special regex symbols in findall.")

    ind = [m.start() for m in finditer(pattern, string)]

    return ind

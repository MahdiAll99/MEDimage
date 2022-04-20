#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List


def getFilePaths(pathToParentFolder, wildcard=None) -> List[Path]:
    """Finds all files in the given path that matches the pattern/wildcard. 

    Note:
        The search is done recursively in all subdirectories.

    Args:
        pathToParentFolder (Path): Full path to where the files are located.
        wildcard (:obj:`str`, optional): String specifying which type of files 
        to locate in the parent folder.
            - Ex : '*.dcm*', to look for dicom files.

    Returns:
        List: List of full paths to files with the specific wildcard located 
            in the given path to parent folder.
    
    """
    if wildcard is None:
        wildcard = '*'

    # Getting the list of all files full path in filePaths
    pathToParentFolder = Path(pathToParentFolder)
    filePaths_list = list(pathToParentFolder.rglob(wildcard))
    # for the name only put file.name
    filePaths = [file for file in filePaths_list if file.is_file()]

    return filePaths

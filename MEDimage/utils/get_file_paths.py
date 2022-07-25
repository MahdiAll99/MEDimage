#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from importlib.resources import path
from pathlib import Path
from typing import List, Union


def get_file_paths(path_to_parent_folder: Union[str, Path], wildcard: str=None) -> List[Path]:
    """Finds all files in the given path that matches the pattern/wildcard. 

    Note:
        The search is done recursively in all subdirectories.

    Args:
        path_to_parent_folder (Union[str, Path]): Full path to where the files are located.
        wildcard (str, optional): String specifying which type of files 
        to locate in the parent folder.
            - Ex : '*.dcm*', to look for dicom files.

    Returns:
        List: List of full paths to files with the specific wildcard located \
            in the given path to parent folder.
    """
    if wildcard is None:
        wildcard = '*'

    # Getting the list of all files full path in file_paths
    path_to_parent_folder = Path(path_to_parent_folder)
    file_paths_list = list(path_to_parent_folder.rglob(wildcard))
    # for the name only put file.name
    file_paths = [file for file in file_paths_list if file.is_file()]

    return file_paths

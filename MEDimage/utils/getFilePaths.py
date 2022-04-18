#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def getFilePaths(pathToParentFolder, wildcard=None):
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

    # IMPORTANT
    # - pathFiles: Full path to where the files are located
    # - wildcard: (optional). String specifying which type of files to locate
    #   in the parent folder.
    #    Ex: '*.dcm*'. If not present --> same as wildcard '*'
    # --> We only want names of files not directories with potentially the
    #     same wildcard name.
    # - The search is done recursively in all subdirectories.

    if wildcard is None:
        wildcard = '*'

    # Getting the list of all files full path in filePaths

    pathToParentFolder = Path(pathToParentFolder)
    filePaths_list = list(pathToParentFolder.rglob(wildcard))
    # for the name only put file.name
    filePaths = [file for file in filePaths_list if file.is_file()]

    return filePaths

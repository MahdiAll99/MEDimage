#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time as t
from pathlib import Path


def waitBatch(pathCheck, time, nBatch):
    """
    -------------------------------------------------------------------------
    DESCRIPTION:
    This function implements a waiting loop ensuring that all the
    computations from all parallel batch are done.
    -------------------------------------------------------------------------
    INPUTS:
    1. pathCheck: Full path to the directory where the 'batch1_end',
      'batch2_end', etc. (parallel checkpoints) are saved.
    2. time: Number of seconds to wait before checking if parallel
       computations are done.
       --> Ex: 60
    3. nBatch: Number of parallel batch.
       --> Ex: 8
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

    startpath = Path.cwd()
    os.chdir(pathCheck)
    list_end_files = ['batch'+str(i)+'_end' for i in range(0, nBatch)]
    list_all_files = os.listdir(".")

    while not all(elem in list_all_files for elem in list_end_files):
        t.sleep(time)
        list_all_files = os.listdir(".")

    os.chdir(startpath)

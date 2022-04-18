#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def batchPatients(nPatient, nBatch):
    """Compute batchPatients.
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

    # FIND THE NUMBER OF PATIENTS IN EACH BATCH
    patients = [0] * nBatch  # np.zeros(nBatch, dtype=int)
    patientVect = np.random.permutation(nPatient)  # To randomize stuff a bit.
    if nBatch:
        nP = nPatient / nBatch
        nSup = np.ceil(nP).astype(int)
        nInf = np.floor(nP).astype(int)
        if nSup != nInf:
            nSubInf = nBatch - 1
            nSubSup = 1
            total = nSubInf*nInf + nSubSup*nSup
            while total != nPatient:
                nSubInf = nSubInf - 1
                nSubSup = nSubSup + 1
                total = nSubInf*nInf + nSubSup*nSup

            nP = np.hstack((np.tile(nInf, (1, nSubInf))[
                           0], np.tile(nSup, (1, nSubSup))[0]))
        else:  # The number of patients in all batches will be the same
            nP = np.tile(nSup, (1, nBatch))[0]

        start = 0
        for i in range(0, nBatch):
            patients[i] = patientVect[start:(start+nP[i])].tolist()
            start += nP[i]

    return patients

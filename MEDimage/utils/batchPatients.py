#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def batchPatients(nPatient, nBatch) -> np.ndarray:
    """Replaces volume intensities outside the ROI with NaN.

    Args:
        nPatient (int): Number of patient.
        nBatch (int): Number of batch, usually less or equal
            to the cores number on your machine.

    Returns:
        ndarray: List of indexes with size nBatch and max value nPatient.
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

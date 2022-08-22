#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def batch_patients(n_patient: int,
                   n_batch: int) -> np.ndarray:
    """Replaces volume intensities outside the ROI with NaN.

    Args:
        n_patient (int): Number of patient.
        n_batch (int): Number of batch, usually less or equal to the cores number on your machine.

    Returns:
        ndarray: List of indexes with size n_batch and max value n_patient.
    """

    # FIND THE NUMBER OF PATIENTS IN EACH BATCH
    patients = [0] * n_batch  # np.zeros(n_batch, dtype=int)
    patient_vect = np.random.permutation(n_patient)  # To randomize stuff a bit.
    if n_batch:
        n_p = n_patient / n_batch
        n_sup = np.ceil(n_p).astype(int)
        n_inf = np.floor(n_p).astype(int)
        if n_sup != n_inf:
            n_sub_inf = n_batch - 1
            n_sub_sup = 1
            total = n_sub_inf*n_inf + n_sub_sup*n_sup
            while total != n_patient:
                n_sub_inf = n_sub_inf - 1
                n_sub_sup = n_sub_sup + 1
                total = n_sub_inf*n_inf + n_sub_sup*n_sup

            n_p = np.hstack((np.tile(n_inf, (1, n_sub_inf))[
                           0], np.tile(n_sup, (1, n_sub_sup))[0]))
        else:  # The number of patients in all batches will be the same
            n_p = np.tile(n_sup, (1, n_batch))[0]

        start = 0
        for i in range(0, n_batch):
            patients[i] = patient_vect[start:(start+n_p[i])].tolist()
            start += n_p[i]

    return patients

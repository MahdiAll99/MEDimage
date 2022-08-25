CSV File
========

In ``MEDimage`` every dataset must have a csv file along with it, this file contains information 
about the scans in the dataset that will be used in the radiomics analysis, especially ROI names used
each scan. Since scans can have multiple ROIs, the user need to specify for each scan tha ROI name(s) to use
for the processing and radiomics extraction. The csv files are also used by the ``DataManager`` in pre-checks and
summary creation (after raw data processing). The different columns of the file are:

- **PatientID**: The scan patient ID, for example:  ``"Glioma-TCGA-001"``.
- **ImagingScanName**: Type of the imaging modality, usually ``"CT"`` for CT scans, ``"CT"`` for PET scans and MRI sequence for MR scans (``"T1"``, ``"T2"``...).
- **ImagingModality**: Imaging modality (``"CTscan"``, ``"MRscan"`` or ``"PTscan"``).
- **ROIname**: ROI name for the analysis. Either addition or subtraction of ROIs or just a single ROI. Every ROI is put between brackets then added or subtracted to other ROIs. For example: ``"{swelling}+{edema}"``.

.. note::
    The csv files must respect the following naming norm:
    ``"roiNames_{roi_label}.csv"``. For example: ``"roiNames_GTV.csv"`` for gross tumor volume. So different patients might have different ROI names,
    but the same roi label.

The following table is an example of a csv for a dataset of Soft-Tissue-Sarcoma (STS) cancer of 3 scans with different modalities (MR, CT and PET) 
and ROIs (GTV mass and GTV edema):

.. list-table:: roiNames_GTV.csv
    :widths: 25 25 25 25
    :header-rows: 1

    *   - PatientID
        - ImagingScanName
        - ImagingModality
        - ROIname
    *   - STS-McGill-001
        - T1
        - MRscan
        - {GTV_Mass}+{GTV_Edema}
    *   - STS-McGill-001
        - T2
        - MRscan
        - {GTV_Edema}-{GTV_Mass}
    *   - STS-McGill-002
        - CT
        - CTscan
        - {GTV_Edema}
    *   - STS-McGill-003
        - PET
        - PTscan
        - {GTV_Mass}

.. note::
    Future works of ``MEDimage`` will aim to automate the creation of these csv files for each dataset and to implement ROIs intersection as well.

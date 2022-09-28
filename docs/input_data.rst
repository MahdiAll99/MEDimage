Input Data
==========

``MEDimage`` package accepts two formats of input data: `NIfTI <https://brainder.org/2012/09/23/the-nifti-file-format/>`__ 
and `DICOM <https://fr.wikipedia.org/wiki/Digital_imaging_and_communications_in_medicine>`__. Each format has its own conventions
that need to be respected. The following sections describe the norms and the conventions for each format and we recommend you process your 
dataset in a way that respects them.

DICOM
-----

Every DICOM file contains a header and a body. The header contains the metadata of the image, and the body contains the image itself.
The header contains a lot of information that helps identify scans from each other, but the most important are the following:

- **Patient ID**: Primary identifier for the Patient, referenced in the ``PatientID`` field of the header. This field should not contain any
  underscore and we recommend that it respects the following format: ``'study-institution-numericID'``. For example, ``'STS-McGill-001'``.
- **Series description**: A description of the series, usually describes the type of the modality used. Referenced in the ``SeriesDescription`` 
  field of the header. This field should be in alphanumeric characters only. For example, ``'T1'``.

Besides the file names, these two fields are used in in the :doc:`../csv_file` of the dataset, in the ``PatientID`` and ``ImagingScanName`` columns that
helps determine scans that will be used in the radiomics analysis.

NIfTI
-----

The NIfTI format is a very simple format that only contains the image itself. Unlike DICOM, the NIfTI format does contain any
information about the regions of interest (ROI) so it needs to be provided in other separate files. In order for ``MEDimage`` to read a NIfTI scan
files, they need to be put in the same folder with the following names:

- ``'PatientID__SeriesDescription(tumorAuto).Modality.nii.gz'``: The image itself. For example: ``'STS-McGill-001__T1(tumorAuto).MRscan.nii.gz'``.
- ``'PatientID__SeriesDescription(ROIname).ROI.nii.gz'``: The ROI of the image. This file should contain a binary mask of the ROI. 
  For example: ``'STS-McGill-001__T1(GTV_Mass).ROI.nii.gz'``.

The following figure sums up the ``MEDimage`` logic in reading data for both formats:

.. image:: /figures/InputDataSummary.png
    :width: 1000
    :align: center

.. note::
    For instance, ``MEDimage`` assumes that all the NIfTI files are ``Axial`` and ``HFS`` so make sure your scans have this same orientation
    and this same patient position. Future works will include the automatic pre-processing of datasets according to the package conventions.

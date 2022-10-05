Input Data
==========

``MEDimage`` package accepts two formats of input data: `NIfTI <https://brainder.org/2012/09/23/the-nifti-file-format/>`__ 
and `DICOM <https://fr.wikipedia.org/wiki/Digital_imaging_and_communications_in_medicine>`__. Each format has its own conventions
that need to be followed. The following sections describe the norms and the conventions for each format and we recommend you process your 
dataset in a way that respects them.

DICOM
-----

Every DICOM file contains a header and a body. The header contains the metadata of the image, and the body contains the image itself.
The header contains a lot of information that helps identify scans from each other, but the most important are the following:

- **Patient ID**: Primary identifier for the Patient, referenced in the ``PatientID`` field of the header. This field should not contain any
  underscore and for compatibility with other `MEDomics packages <https://github.com/medomics>`__, we recommend that it respects the following 
  format: ``'study-institution-numericID'``. For example, ``'STS-McGill-001'``. It is also used in the :doc:`../csv_file` of the dataset under 
  the column ``PatientID``.
- **Series description**: Referenced in the ``SeriesDescription`` field of the DICOM header. A description of the series, usually describes the 
  type of the modality used or gives it a name to help differentiate scans with the same modality (Imaging scan name). It should be in alphanumeric characters 
  only. For example, ``'T1'`` or ``'T2'`` for ``'MRI'`` scans. It is also referred to in the :doc:`../csv_file` of the dataset as ``ImagingScanName``.

NIfTI
-----

The NIfTI format is a simple format that only contains the image itself. Unlike DICOM, the NIfTI format does contain any
information about the regions of interest (ROI) so it needs to be provided in other separate files. In order for ``MEDimage`` to read a NIfTI scan
files, they need to be put in the same folder with the following names:

- ``'PatientID__SeriesDescription(ROILabel).Modality.nii.gz'``: The image itself. For example: ``'STS-McGill-001__T1(GTV).MRscan.nii.gz'``.
- ``'PatientID__SeriesDescription(ROIname).ROI.nii.gz'``: The ROI or the mask of the image. This file should contain a binary mask of the ROI. 
  For example: ``'STS-McGill-001__T1(GTV_Mass).ROI.nii.gz'``.

The following figure sums up the ``MEDimage`` logic in reading data for both formats:

.. image:: /figures/InputDataSummary.png
    :width: 1000
    :align: center

If these conventions are followed, the ``DataManager`` class will be able to read the data and create the ``MEDscan`` objects that will be used
in the radiomics analysis. Furthermore, we suggest you organize your dataset folder as follows:

::

    dataset_folder
    ├── Patient ID 1      
    │   ├── ImagingScanName 1
    │   │   ├── DICOM files
    │   │   └── ...
    │   └── ImagingScanName 2
    │       ├── DICOM files
    │       └── ...
    ├── Patient ID 2      
    │   ├── ImagingScanName 1
    │   │   ├── DICOM files
    │   │   └── ...
    │   └── ImagingScanName 2
    │       ├── DICOM files
    │       └── ...
    └── ...

For example:

::

    dataset_folder
    ├── STS-McGill-001      
    │   ├── T1
    │   │   ├── *.dcm
    │   │   └── ...
    │   └── PET
    │       ├── *.dcm
    │       └── ...
    ├── STS-McGill-002      
    │   ├── T2FS
    │   │   ├── *.dcm
    │   │   └── ...
    │   └── CT
    │       ├── *.dcm
    │       └── ...
    └── ...

.. note::
    For instance, ``MEDimage`` assumes that all the NIfTI files are ``Axial`` and ``HFS`` so make sure your scans have this same orientation
    and this same patient position. Future works will include the automatic pre-processing of datasets according to the package conventions.

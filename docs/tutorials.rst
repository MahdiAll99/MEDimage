Instructions
============

Download dataset
----------------
    In all tutorials, we use an open-access dataset containing medical images for various cancer types (Glioma, sarcoma...) 
    and different imaging modalities (MR, CT, and PET). The dataset has been pre-processed to adhere to package norms.

    To download the dataset (~3.2 GB) and organize it in your local workspace, run the following command in your terminal from 
    the package parent folder ::
    
       python scripts/download_data.py --full-sts
    
    .. note::
        The dataset is large, and options are available to download only a subset. For more information, run:
        
           python scripts/download_data.py --help

CSV file
--------

    Most tutorials, such as the :ref:`BatchExtractor tutorial <BatchExtractor>`, utilize multiple scans, each with its CSV file. 
    ``MEDimage`` requires a CSV file for each dataset; details can be found in the :doc:`../csv_file`. 
    Examples are available in ``MEDimage/notebooks/tutorial/csv``.

    .. note::
        Future versions of ``MEDimage`` aim to automate the creation of these CSV files for each dataset.

Configuration file
------------------

    To use ``MEDimage``, a configuration file is always required. An example file is available in the GitHub repository
    (``MEDimage/notebooks/tutorial/settings/MEDimage-Tutorial.json``), and documentation is provided :doc:`../configurations_file`.
    Different JSON configuration files are used for each case; for example, specific JSON configurations for every
    `IBSI <https://theibsi.github.io/>`__ test are available in ``MEDimage/notebooks/ibsi/settings``.

DataManager
===========

    The ``DataManager`` plays an important role in ``MEDimage``. The class is capable of processing raw `DICOM <https://en.wikipedia.org/wiki/DICOM>`__ 
    and `NIfTI <https://brainder.org/2012/09/23/the-nifti-file-format/>`__ and converting them in into ``MEDscan`` class objects. It includes pre-radiomics 
    analysis, determining the best intensity ranges and voxel dimension rescaling parameters for a given dataset.
    This analysis is essential, as highlighted in this `article <https://doi.org/10.1016/j.ejmp.2021.07.023>`__ , which investigates how intensity 
    window settings can impact radiomic feature stability for CT data.
    
    The tutorial for DataManager is available here.: |DataManager_image_badge|

    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/DataManager-Tutorial.ipynb``.

.. |DataManager_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/DataManager-Tutorial.ipynb

.. image:: /figures/DataManager-overview.png
    :width: 800
    :align: center

MEDscan Class
==============

    In MEDimage, the ``MEDscan`` class is a Python object that maintains data and information about the dataset, particularly related to scans processed 
    from NIfTI or DICOM data. It can manage parameters used in processing, filtering, and extraction, reading from JSON files and updating all relevant 
    attributes. Many other useful functionalities are detailed in this tutorial: |MEDimage_image_badge|
    
    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/MEDimage-Tutorial.ipynb``.

.. |MEDimage_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/MEDimage-Tutorial.ipynb

Single-scan demo
================

    This demo provides a step-by-step guide to processing and extracting features for a single scan using ``MEDimage``. It covers various use cases, 
    from initial processing steps to the extraction of features. The demo is perfect for learning how to use MEDimage for single-scan feature extraction.
    
    The interactive Colab notebook for the demo is available here: |Glioma_demo_image_badge|

    You can also find it on the repository ``MEDimage/notebooks/demo/Glioma-Demo.ipynb``.

.. |Glioma_demo_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/demo/Glioma-Demo.ipynb

BatchExtractor
==============

    ``MEDimage`` facilitates batch feature extraction through the ``BatchExtractor`` class, which streamlines the following workflow: 

    .. image:: /figures/BatchExtractor-overview.png
        :width: 800
        :align: center
    
    This class creates batches of scans and performs full extraction of all radiomics family features, saving them in tables and JSON files. 
    To run a batch extraction, simply set the path to your dataset and the path to your dataset's :doc:`../csv_file` of regions of interest.
    (check example `here <https://github.com/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/CSV/roiNames_GTV.csv>`__).

    Learn more in the interactive Colab notebook here: |BatchExtractor_image_badge|
    
    You can also find it on the repository ``MEDimage/notebooks/tutorial/BatchExtractor-Tutorial.ipynb``.

.. |BatchExtractor_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/BatchExtractor-Tutorial.ipynb

Learning
========

    ``MEDimage`` offers a learning module for training a machine learning model on extracted features. The module handles features cleaning, normalization, 
    selection, model training, and testing. The workflow is summarized in the following image:

    .. image:: /figures/LearningWorkflow.png
        :width: 800
        :align: center

    Similar to the extraction module, the learning module also uses multiple JSON configuration files to set the parameters of the learning process.
    Details about the configuration files, are available here: :doc:`../configurations_file`. You can also find an example of these files in the 
    GitHub repository (``MEDimage/tree/learning/notebooks/tutorial/learning/settings``).
    
    A tutorial is provided in this notebook: |Learning_image_badge|

    You can also find it on the repository ``MEDimage/notebooks/tutorial/Learning-Tutorial.ipynb``.

.. |Learning_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/learning/notebooks/tutorial/Learning-Tutorial.ipynb
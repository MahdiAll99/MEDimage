Instructions
============

Download dataset
----------------
    In all the tutorials, an open-access dataset will be used. It consists of medical images for different types of cancer (Glioma, sarcoma...)
    and with different imaging modalities (MR, CT and PET). This dataset has been pre-processed in order to be compliant with the package norms.

    A script is made available to download the dataset (~3.2 GB) and organize it in your local workspace, just run the following command in your terminal
    from the  package parent folder ::
    
       python scripts/download_data.py --full-sts
    
    .. note::
        Since the dataset is large, there are available options to download only a subset of the data. For more information, run the following command in your terminal ::
        
           python scripts/download_data.py --help

CSV file
--------

    In most tutorials (:ref:`BatchExtractor tutorial <BatchExtractor>` for example) that use multiple scans, you will notice 
    the use of different csv files depending on the datasets. ``MEDimage`` requires that every dataset must have a csv file along with it, 
    we recommend taking a look into the documentation in :doc:`../csv_file`. You can also check some examples in 
    ``MEDimage/notebooks/tutorial/csv``.

    .. note::
        Future works of ``MEDimage`` will aim to automate the creation of these csv files for each datasets.

Configuration file
------------------

    In order to use ``MEDimage``, you will always need a configuration file, you can find an example of this file in the GitHub repository
    (``MEDimage/notebooks/tutorial/settings/MEDimage-Tutorial.json``) and the documentation is available in :doc:`../configuration_file`.
    And for each case, we will use a different JSON configuration file. For example every `IBSI <https://theibsi.github.io/>`__
    test requires specific JSON configuration and you can find all of them in: ``MEDimage/notebooks/ibsi/settings``.

DataManager
===========

    The ``DataManager`` plays an important role in ``MEDimage``. The class is capable of processing raw `DICOM <https://en.wikipedia.org/wiki/DICOM>`__ 
    and `NIfTI <https://brainder.org/2012/09/23/the-nifti-file-format/>`__ and converting them in into ``MEDscan`` class objects. This class also offers
    a pre-radiomics analysis which consists of finding the best intensity ranges and best voxel dimension rescaling parameters 
    for a given dataset and its :doc:`../csv_file`, since these options impacts the radiomics analysis. For example, this 
    `article <https://doi.org/10.1016/j.ejmp.2021.07.023>`__ investigates how intensities window can impact the radiomic features stability for CT data.
    
    The tutorial is an interactive Colab notebook and is directly accessible here: |DataManager_image_badge|

    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/DataManager-Tutorial.ipynb``.

.. |DataManager_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/DataManager-Tutorial.ipynb

.. image:: /figures/DataManager-overview.png
    :width: 800
    :align: center

MEDscan Class
==============

    In the ``MEDimage`` package, we have a class named ``MEDscan`` which is a Python object that maintains data and information about the dataset, 
    related to the scans processed from NIfTI or DICOM data. The ``MEDscan`` class is also capable of managing the parameters used in processing, 
    filtering and extraction. It can read JSON files and update all the parameters related attributes in the class. This class offers many other 
    useful functionalities that you can find out about in the interactive Colab notebook here: |MEDimage_image_badge|
    
    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/MEDimage-Tutorial.ipynb``.

.. |MEDimage_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/MEDimage-Tutorial.ipynb

Single-scan demo
================

    This demo is a step by step guide to process and extract features for a single scan using ``MEDimage``. We try in this demo to cover all the possible
    use cases of the package and its subpackages from the first steps of processing until the last steps of features extraction. we process the scan,
    initialize the ``MEDscan`` class, process the imaging data and extract features. So this demo is perfect to learn how to use ``MEDimage`` for single
    scan features extraction.
    
    The demo is an interactive Colab notebook and is directly accessible here: |Glioma_demo_image_badge|

    You can also find this demo on the repository ``MEDimage/notebooks/demo/Glioma-Demo.ipynb``.

.. |Glioma_demo_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/demo/Glioma-Demo.ipynb

BatchExtractor
==============

    ``MEDimage`` allows batch features extraction through the class ``BatchExtractor`` which is a simple Python class with the following workflow:

    .. image:: /figures/BatchExtractor-overview.png
        :width: 800
        :align: center
    
    It is capable of creating batches of scans with not so many arguments and running a full extraction of all the radiomics family features and saving
    it in tables and JSON files. In order to run a batch extraction using this class, you will only need to set the path to your dataset and to your 
    dataset :doc:`../csv_file` of the regions of interest (check example `here <https://github.com/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/CSV/roiNames_GTV.csv>`__).
    
    This class is made very easy to use and we recommend you check this tutorial in the interactive Colab notebook here: |BatchExtractor_image_badge|
    
    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/BatchExtractor-Tutorial.ipynb``.

.. |BatchExtractor_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/BatchExtractor-Tutorial.ipynb

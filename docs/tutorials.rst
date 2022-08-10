Download dataset
----------------

    In all the tutorials, an open-access dataset will be used. It consists of medical images for different type of cancers (Glioma, sarcoma...)
    and with different imaging modalities (MR, CT...). This dataset has been pre-processed in order to be compliant with the package norms.

    To download the dataset (~3.1GB) and organize it in your local workspace, run the following command in your terminal::
    
       python download_organize_data.py

Configuration file
------------------

    In order to use ``MEDimage``, you will always need a configuration file, you can find examples of these files in the GitHub repository
    (``MEDimage/notebooks/tutorial/settings/MEDimage-Tutorial.json``) and the documentation is available in :doc:`../configuration_file`.
    And for each case, we will use a different JSON configuration file. For example every `IBSI <https://theibsi.github.io/>`__
    test requires specific JSON configurations that you can find in ``MEDimage/notebooks/ibsi/settings``.

DataManager
-----------

    The ``DataManager`` plays an important role in ``MEDimage``. The class is capable of processing raw `DICOM <https://en.wikipedia.org/wiki/DICOM>`__ 
    and `NIfTI <https://brainder.org/2012/09/23/the-nifti-file-format/>`__ and converting in into ``MEDimage`` class objects. This class also offers
    a pre-radiomics analysis which consists of finding the best intensity ranges and best voxel dimension rescaling parameters, since these options
    impacts the radiomics features, you can read more about this in this `article <https://doi.org/10.1016/j.ejmp.2021.07.023>`__.
    
    The tutorial is an interactive Colab notebook and is directly accessible here: |DataManager_image_badge|

    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/DataManager-Tutorial.ipynb``.

.. |DataManager_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/DataManager-Tutorial.ipynb

MEDimage Class
--------------

    In ``MEDimage``, we have the package ``MEDimage`` and  the ``MEDimage`` class which is a Python object with the following structure:

    .. image:: https://github.com/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/images/MEDimageClassDiagram.png?raw=true
        :alt: MEDimage class diagram

    
    It maintains data and information extracted related to the scans processed from NIfTI or DICOM data. The ``MEDimage`` class is also capable 
    of managing the parameters used in processing, filtering and extraction. It can read JSON files and update all the parameters related attributes 
    in the class. This class offers many other useful functionalities that you can find out about in the interactive Colab notebook here: |MEDimage_image_badge|
    
    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/MEDimage-Tutorial.ipynb``.

.. |MEDimage_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/MEDimage-Tutorial.ipynb

Single-scan demo
----------------

    This demo is a step by step guide to process and extract features for a single scan using ``MEDimage``. We try in this demo to cover all the possible
    use cases of the package and its subpackages from the first steps of processing until the last steps of features extraction. we process the scan,
    initialize the ``MEDimage`` class, process the imaging data and extract features. So this demo is perfect to learn how to use ``MEDimage`` for single
    scan features extraction.
    
    The demo is an interactive Colab notebook and is directly accessible here: |Glioma_demo_image_badge|

    You can also find this demo on the repository ``MEDimage/notebooks/demo/Glioma-Demo.ipynb``.

.. |Glioma_demo_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/demo/Glioma-Demo.ipynb

BatchExtractor
--------------

    ``MEDimage`` allows batch features extraction, using the class ``BatchExtractor`` which is a Python class with that has the following simple structure:

    .. image:: https://github.com/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/images/BatchExtractionClassDiagram.png?raw=true
        :alt: MEDimage class diagram

    
    It is capable of creating batches of scans with not so many arguments and running a full extraction of all the radiomics family features and saving
    it in tables and JSON files. In order to run a batch extraction using this class, you will only need to set the path to your dataset and to your 
    dataset CSV file of the regions of interest (check example `here <https://github.com/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/CSV/roiNames_GTV.csv>`__).
    
    This class is made very easy to use and we recommend you check this tutorial in the interactive Colab notebook here: |BatchExtractor_image_badge|
    
    You can also find this tutorial on the repository ``MEDimage/notebooks/tutorial/BatchExtractor-Tutorial.ipynb``.

.. |BatchExtractor_image_badge| image:: https://colab.research.google.com/assets/colab-badge.png
    :target: https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/BatchExtractor-Tutorial.ipynb
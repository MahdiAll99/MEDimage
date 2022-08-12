Configuration File
==================

In ``MEDimage``, all the subpackages and modules need a specific configuration to be used correctly, so they respectively
rely on one single JSON configuration file. This file contains parameters for each step of the workflow (processing, extraction...).
For example, `IBSI <https://arxiv.org/abs/1612.07003>`__ tests require specific parameters for radiomcs extraction for each test.
You can check a full example of the file here: 
`notebooks/ibsi/settings/ <https://github.com/MahdiAll99/MEDimage/tree/main/notebooks/ibsi/settings>`__.

This section will go through the details of how to set up and use this configuration file and will be separated to four subdivision:

- :ref:`Pre-checks<Pre-checks Parameters>`
- :ref:`Processing<Processing Parameters>`
- :ref:`Radiomics<Extraction Parameters>`
- :ref:`Filters<Filtering Parameters>`

Pre-checks Parameters
---------------------
The pre radiomics checks configuration is a set of parameters used by the ``DataManager`` class. These parameters must be set in a nested
dictionary as follows:

.. code-block:: JSON

    {
        "pre_radiomics_checks": {"All parameters go inside this dict"}
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "wildcards_dimensions",
        "description": "List of wild cards for voxel dimension checks (Read about wildcards
             `here <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`__), 
             checks will be done for every wildcard in the list. For example ``[\"Glioma*.MRscan.npy\", \"STS*.CTscan.npy\"]``",
        "type": "List[str]"
    }


.. code-block:: JSON

    {
        "pre_radiomics_checks" : {
            "wildcards_dimensions" : ["Glioma*.MRscan.npy", "STS*.CTscan.npy"],
            }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "wildcards_window",
        "description": "List of wild cards for intensities window checks (Read about wildcards
             `here <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`__), 
             checks will be done for every wildcard in the list. For example ``[\"Glioma*.MRscan.npy\", \"STS*.CTscan.npy\"]``",
        "type": "List[str]"
    }


.. code-block:: JSON

    {
        "pre_radiomics_checks" : {
            "wildcards_window" : ["Glioma*.MRscan.npy", "STS*.CTscan.npy"],
            }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "path_data",
        "description": "Path to your data (``MEDimage`` class pickle objects)",
        "type": "str"
    }


.. code-block:: JSON

    {
        "pre_radiomics_checks" : {
            "path_data" : "home/user/medimage/data/npy/sts",
            }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "path_csv",
        "description": "Path to your dataset csv file (Read more about the :doc:`../csv_file`)",
        "type": "str"
    }


.. code-block:: JSON

    {
        "pre_radiomics_checks" : {
            "path_csv" : "home/user/medimage/data/csv/roiNames_GTV.csv",
            }
    }

.. note::
    initializing the :ref:`pre-radiomics checks settings<Pre-checks Parameters>` 
    is optional and can be done later while using the ``DataManager`` instance.

Processing Parameters
---------------------

Each imaging modality should have its own params dict inside the JSON file and should be organized as follows:

.. code-block:: JSON

    {
        "imParamMR": {"Processing parameters for MR modality"},
        "imParamCT": {"Processing parameters for CT modality"},
        "imParamPET": {"Processing parameters for PET modality"}
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "box_string",
        "description": "Box of the ROI used in the workflow.",
        "type": "string",
        "options": {
            "full": {
                "description": "Use the full ROI",
                "type": "string"
            },
            "box": {
                "description": "Use the smallest box possible",
                "type": "string"
            },
            "box{n}": {
                "description": "For example ``box10``, 10 voxels are added in all three dimensions
                    the smallest bounding box. The number after 'box' defines the number of voxels to add.",
                "type": "string"
            },
            "{n}box": {
                "description": "For example ``2box``, Will use double the size of the smallest box . 
                    The number before 'box' defines the multiplication in size.",
                "type": "string"
            }
        }
    }


.. code-block:: JSON

    {
        "imParamCT" : {
            "box_string" : "box7",
            },
        "imParamMR" : {
            "box_string" : "box",
            },
        "imParamPET" : {
            "box_string" : "2box",
            },
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "interp",
        "description": "Interpolation parameters.",
        "type": "dict",
        "options": {"scale_non_text": {
                        "description": "size-3 list of the new voxel size",
                        "type": "List[float]"
                    },
                    "scale_text": {
                        "description": "Lists of size-3 of the new voxel size for texture features (features will be computed for each list)",
                        "type": "List[List[float]]"
                    },
                    "vol_interp": {
                        "description": "Volume interpolation method (\"linear\", \"spline\" or \"cubic\")",
                        "type": "string"
                    },
                    "gl_round": {
                        "description": "This option should be set only for CT scans, set it to 1 to round values to nearest integers 
                            (Must be a power of 10)",
                        "type": "float"
                    },
                    "roi_interp": {
                        "description": "ROI interpolation method (\"nearest\", \"linear\" or \"cubic\")",
                        "type": "string"
                    },
                    "roi_pv": {
                        "description": "Rounding value for ROI intensities. Must be between 0 and 1.",
                        "type": "float"
                    }
        }
    }


.. code-block:: JSON

    {
        "imParamCT" : {
            "interp" : {
                "scale_non_text" : [2, 2, 3],
                "scale_text" : [[2, 2, 3]],
                "vol_interp" : "linear",
                "gl_round" : 1,
                "roi_interp" : "linear",
                "roi_pv" : 0.5
            },
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "reSeg",
        "description": "Resegmentation parameters.",
        "type": "dict",
        "options": {
                    "range": {
                        "description": "Resegmentation range, 2-elements list consists of minimum and maximum intensity value. Use ``\"inf\"`` for
                        infinity",
                        "type": "List"
                    },
                    "outliers": {
                        "description": "Outlier resegmentation algorithm. For now ``MEDimage`` only implements ``\"Collewet\"`` algorithms.
                            Leave empty for no outlier resegmentation",
                        "type": "string"
                    }
        }
    }


.. code-block:: JSON

    {
        {
        "imParamCT" : {
            "reSeg" : {
                "range" : [-500, "inf"],
                "outliers" : ""
            },
        },
        {
        "imParamMR" : {
            "reSeg" : {
                "range" : [-500, 500],
                "outliers" : "Collewet"
            },
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "discretisation",
        "description": "Discretisation parameters.",
        "type": "dict",
        "options": {
                    "IH": {
                        "description": "Discretisation parameters for intensity histogram features",
                        "type": "dict"
                    },
                    "IVH": {
                        "description": "Discretisation parameters for intensity volume histogram features",
                        "type": "dict"
                    },
                    "texture": {
                        "description": "Discretisation parameters for texture features",
                        "type": "dict"
                    }
        }
    }

- **IH**

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "Discretisation parameters for intensity histogram features.",
        "type": "dict",
        "options": {
                    "type": {
                        "description": "Discretisation algorithm: ``\"FBS\"`` for fixed bin size and
                            ``\"FBN\"`` for fixed bin number algorithm",
                        "type": "string"
                    },
                    "val": {
                        "description": "Bin size or bin number, depending on the algorithm used",
                        "type": "int"
                    }
        }
    }

- **IVH**

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "Discretisation parameters for intensity volume histogram features.",
        "type": "dict",
        "options": {
                    "type": {
                        "description": "Discretisation algorithm: ``\"FBS\"`` for fixed bin size and
                            ``\"FBN\"`` for fixed bin number algorithm",
                        "type": "string"
                    },
                    "val": {
                        "description": "Bin size or bin number, depending on the algorithm used",
                        "type": "int"
                    }
        }
    }

- **texture**

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "Discretisation parameters for texture features.",
        "type": "dict",
        "options": {
                    "type": {
                        "description": "List of discretisation algorithms: ``\"FBS\"`` for fixed bin size and
                            ``\"FBN\"`` for fixed bin number. Texture features will be computed for each algorithm in the list",
                        "type": "List[string]"
                    },
                    "val": {
                        "description": "List of bin sizes or bin numbers, depending on the algorithm used.
                             Texture features will be computed for each bin number or bin size in the list",
                        "type": "List[List[int]]"
                    }
        }
    }


.. code-block:: JSON

    {
        {
        "imParamCT" : {
            "IH" : {
                "type" : "FBS",
                "val" : 25
            },
            "IVH" : {
                "type" : "FBN",
                "val" : 10
            },
            "texture" : {
                "type" : ["FBS", "FBN"],
                "val" : [[25], [10]]
            }
        },
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "compute_suv_map",
        "description": "Computation of the `suv <https://en.wikipedia.org/wiki/Standardized_uptake_value>`__ map for PET scans. Defalt ``True``",
        "type": "bool",
        "options": {
            "True": {
                "description": "Will compute suv map for PET scans.",
                "type": "bool"
            },
            "False": {
                "description": "Will not compute suv map and it must be computed before.",
                "type": "bool"
            }
        }
    }

.. code-block:: JSON

    {
        "imParamPET" : {
            "compute_suv_map" : true
            },
    }

.. note::
   ``MEDimage`` only computes suv map for DICOM scans, since the computation relies on DICOM headers for computation
   and assumes it's already computed for NIfTI scans.


Extraction Parameters
---------------------

Extraction parameters are organized in the same wat as the processing parameters so each imaging modality should have its own parameters and the JSON file should be organized as follows:

.. code-block:: JSON

    {
        "imParamMR": {"Extraction params for MR modality"},
        "imParamCT": {"Extraction params for CT modality"},
        "imParamPET": {"Extraction params for PET modality"}
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "GLCM features distance norm option. by default ``False``",
        "title": "glcm distance_norm",
        "type": "dict",
        "options": {
                    "manhattan": {
                        "description": "Will use ``\"manhattan\"`` weighting norm.",
                        "type": "string"
                    },
                    "euclidean": {
                        "description": "Will use ``\"euclidean\"`` weighting norm.",
                        "type": "string"
                    },
                    "chebyshev": {
                        "description": "Will use ``\"chebyshev\"`` weighting norm.",
                        "type": "string"
                    },
                    "True": {
                        "description": "``True`` in order to usediscretization length difference corrections as used by the 
                            `Institute of Physics andEngineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.
                            Set it to ``False`` to replicate IBSI results.",
                        "type": "bool"
                    },
                    "False": {
                        "description": "``False`` to replicate IBSI results.",
                        "type": "bool"
                    } 
        }
    }

.. code-block:: JSON

    {
        {
        "imParamCT" : {
            "glcm" : {
                "distance_norm" : "chebyshev"
        },
        {
        "imParamMR" : {
            "glcm" : {
                "distance_norm" : false
        },
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "NGTDM features distance norm option. by default ``False``",
        "title": "ngtdm distance_norm",
        "type": "dict",
        "options": {
                    "True": {
                        "description": "``True`` in order to use discretization length difference corrections as used by the 
                            `Institute of Physics andEngineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.
                            Set it to ``False`` to replicate IBSI results.",
                        "type": "bool"
                    },
                    "False": {
                        "description": "``False`` to replicate IBSI results.",
                        "type": "bool"
                    } 
        }
    }

.. code-block:: JSON

    {
        {
        "imParamCT" : {
            "ngtdm" : {
                "distance_norm" : true,
        },
        {
        "imParamMR" : {
            "ngtdm" : {
                "distance_norm" : false
        },
    }


Filtering parameters
--------------------

Filtering parameters are organized  in a separate dictionary, each dictionary contains 
parameters for every filter of the ``MEDimage``:

.. code-block:: JSON

    {
        "imParamFilter": {
            "filter_type": "name of the filter to use",
            "mean": {"mean filter params"},
            "log": {"log filter params"},
            "laws": {"laws filter params"},
            "gabor": {"gabor filter params"},
            "wavelet": {"wavelet filter params"},
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "filter_type",
        "description": "Name of the filter that will be used. Leave empty for no filtering",
        "type": "string",
        "options": {
            "mean": {
                "description": "Use mean filter.",
                "type": "string"
            },
            "log": {
                "description": "Use log filter.",
                "type": "string"
            },
            "laws": {
                "description": "Use laws filter.",
                "type": "string"
            },
            "gabor": {
                "description": "Use gabor filter.",
                "type": "string"
            }
        }
    }

.. code-block:: JSON

    {
        {
        "imParamFilter" : {
            "filter_type" : "laws"
        },
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "mean",
        "description": "Parameters of the mean filter",
        "type": "dict",
        "options": {
            "ndims": {
                "description": "Dimension of the imaging data. Usually 3.",
                "type": "int"
            },
            "size": {
                "description": "Size of the filter kernel.",
                "type": "int"
            },
            "padding": {
                "description": "Padding mode, default ``\"symmetric\"``. All the padding modes possible can be found 
                    `here <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__ ",
                "type": "string"
            },
            "name_save": {
                "description": "Saving name added to the end of every radiomics extraction results table 
                    (Only if the filter was applied).",
                "type": "string"
            }
        }
    }

.. code-block:: JSON

    {
        {
        "imParamFilter" : {
            "mean" : {
                "ndims" : 3,
                "size" : 5,
                "padding" : "symmetric",
                "name_save" : "mean5"
            },
        },
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "log",
        "description": "Parameters of the laplacian of Gaussian filter",
        "type": "dict",
        "options": {
            "ndims": {
                "description": "Dimension of the imaging data. Usually 3.",
                "type": "int"
            },
            "sigma": {
                "description": "Standard deviation of the Gaussian, controls the scale of the convolutional operator.",
                "type": "float"
            },
            "orthogonal_rot": {
                "description": "If ``True``, the images will be rotated over all the planes.",
                "type": "bool"
            },
            "padding": {
                "description": "Padding mode, default ``\"symmetric\"``. All the padding modes possible can be found 
                    `here <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__ ",
                "type": "string"
            },
            "name_save": {
                "description": "Saving name added to the end of every radiomics extraction results table 
                    (Only if the filter was applied).",
                "type": "string"
            }
        }
    }

.. code-block:: JSON

    {
        {
        "imParamFilter" : {
            "log" : {
                "ndims" : 3,
                "sigma" : 1.5,
                "orthogonal_rot" : false,
                "padding" : "constant",
                "name_save" : "log_1.5"
            },
        },
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "laws",
        "description": "Parameters of the laws filter",
        "type": "dict",
        "options": {
            "config": {
                "description": "List of string of every 1D filter used to create the Laws kernel. Possible 1D filters:
                    ``\"L3\"``, ``\"L5\"``, ``\"E3\"``, ``\"E5\"``, ``\"S3\"``, 
                    ``\"S5\"``, ``\"W5\"`` or ``\"R5\"``",
                "type": "List[str]"
            },
            "energy_distance": {
                "description": "The Chebyshev distance that will be used to create the laws texture energy image.",
                "type": "float"
            },
            "rot_invariance": {
                "description": "If ``True``, rotational invariance will be approximated.",
                "type": "bool"
            },
            "orthogonal_rot": {
                "description": "If ``True``, the images will be rotated over all the planes.",
                "type": "bool"
            },
            "energy_image": {
                "description": "If ``True``, Laws texture energy images are computed.",
                "type": "bool"
            },
            "padding": {
                "description": "Padding mode, default ``\"symmetric\"``. All the padding modes possible can be found 
                    `here <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__ ",
                "type": "string"
            },
            "name_save": {
                "description": "Saving name added to the end of every radiomics extraction results table 
                    (Only if the filter was applied).",
                "type": "string"
            }
        }
    }

.. code-block:: JSON

    {
        {
        "imParamFilter" : {
            "laws" : {
                "config" : ["L5", "E5", "E5"],
                "energy_distance" : 7,
                "rot_invariance" : true,
                "orthogonal_rot" : false,
                "energy_image" : true,
                "padding" : "symmetric",
                "name_save" : "laws_l5_e5_e5_7"
            },
        },
    }

.. note::
    The order of the 1D filters used in laws filter configuration matter, because we use the configuration list to compute the outer 
    product and the outer product is not commutative.

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "gabor",
        "description": "Parameters of the gabor filter",
        "type": "dict",
        "options": {
            "sigma": {
                "description": "Standard deviation of the Gaussian envelope, controls the scale of the filter.",
                "type": "float"
            },
            "lambda": {
                "description": "Wavelength or inverse of the frequency.",
                "type": "float"
            },
            "gamma": {
                "description": "Spatial aspect ratio.",
                "type": "float"
            },
            "theta": {
                "description": "Angle of the rotation matrix.",
                "type": "str"
            },
            "rot_invariance": {
                "description": "If ``True``, rotational invariance will be approximated by combining the response 
                    maps of several elements of the Gabor filterbank.",
                "type": "bool"
            },
            "orthogonal_rot": {
                "description": "If ``True``, the images will be rotated over all the planes.",
                "type": "bool"
            },
            "padding": {
                "description": "Padding mode, default ``\"symmetric\"``. All the padding modes possible can be found 
                    `here <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__ ",
                "type": "string"
            },
            "name_save": {
                "description": "Saving name added to the end of every radiomics extraction results table 
                    (Only if the filter was applied).",
                "type": "string"
            }
        }
    }

.. code-block:: JSON

    {
        {
        "imParamFilter" : {
            "gabor" : {
                "sigma" : 5,
                "lambda" : 2,
                "gamma" : 1.5,
                "theta" : "Pi/8",
                "rot_invariance" : true,
                "orthogonal_rot" : true,
                "padding" : "symmetric",
                "name_save" : "gabor_5_2_1.5"
            },
        },
    }

.. note::
    ``gamma`` parameter should be radian but must be specified as a string, for example :math:`\frac{\pi}{2}`
    should be specified as "Pi/2".

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "wavelet",
        "description": "Parameters of the gabor filter",
        "type": "dict",
        "options": {
            "ndims": {
                "description": "Dimension of the imaging data. Usually 3.",
                "type": "int"
            },
            "basis_function": {
                "description": "Wavelet name used to create the kernel.",
                "type": "string"
            },
            "subband": {
                "description": "String of the 1D wavelet kernels (``\"H\"`` for high-pass filter or ``\"L\"`` 
                    for low-pass filter). Must have a size of ``ndims``.",
                "type": "string"
            },
            "level": {
                "description": "The number of decomposition steps to perform.",
                "type": "int"
            },
            "rot_invariance": {
                "description": "If ``True``, rotational invariance will be approximated.",
                "type": "bool"
            },
            "padding": {
                "description": "Padding mode, default ``\"symmetric\"``. All the padding modes possible can be found 
                    `here <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__ ",
                "type": "string"
            },
            "name_save": {
                "description": "Saving name added to the end of every radiomics extraction results table 
                    (Only if the filter was applied).",
                "type": "string"
            }
        }
    }

.. code-block:: JSON

    {
        {
        "imParamFilter" : {
            "wavelet" : {
                "ndims" : 3,
                "basis_function" : "db3",
                "subband" : "LLH",
                "level" : 1,
                "rot_invariance" : true,
                "padding" : "symmetric",
                "name_save" : "Wavelet_db3_LLH"
            },
        },
    }

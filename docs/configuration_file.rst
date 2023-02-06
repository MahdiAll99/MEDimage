Configuration File
==================

In ``MEDimage``, all the subpackages and modules need a specific configuration to be used correctly, so they respectively
rely on one single JSON configuration file. This file contains parameters for each step of the workflow (processing, extraction...).
For example, `IBSI <https://arxiv.org/abs/1612.07003>`__ tests require specific parameters for radiomics extraction for each test.
You can check a full example of the file here: 
`notebooks/ibsi/settings/ <https://github.com/MahdiAll99/MEDimage/tree/main/notebooks/ibsi/settings>`__.

This section will walk you through the details on how to set up and use the configuration file. It will be separated to four subdivision:

- :ref:`Pre-checks<Pre-checks Parameters>`
- :ref:`Processing<Processing Parameters>`
- :ref:`Radiomics<Extraction Parameters>`
- :ref:`Filters<Filtering Parameters>`

General analysis Parameters
---------------------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "n_batch",
        "description": "A numerical value that determines the number of batches to be used in parallel computations, 
            set to 0 for serial computation.",
        "type": "int"
    }

e.g.

.. code-block:: JSON

    {
        "n_batch" : 8
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "roi_type_labels",
        "description": "A list of labels for the regions of interest (ROI) to use in the analysis. The labels must match the names 
            of the corresponding CSV files. For example, if you have a csv file named ``roiNames_GTV.csv``, 
            then the ``roi_type_labels`` msut be ``[\"GTV\"]``.",
        "type": "List[str]"
    }

e.g.

.. code-block:: JSON

    {
        "roi_type_labels" : ["GTV"]
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "roi_types",
        "description": "A list of labels that describe the regions of interest, used to save the analysis results. The labels must accurately 
            reflect the regions analyzed. For instance, if you conduct an analysis of a single ROI in a  ``\"GTV\"`` area with
            two different ROIs (``\"Mass\"`` and ``\"Edema\"``), the label can be ``[\"GTVMassOnly\"]``. This name will be displayed in the
            JSON results file.",
        "type": "List[str]"
    }

e.g.

.. code-block:: JSON

    {
        "roi_types" : ["GTVMassOnly"]
    }



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
             `here <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`__).
             Checks will be run for every wildcard in the list. For example ``[\"Glioma*.MRscan.npy\", \"STS*.CTscan.npy\"]``",
        "type": "List[str]"
    }

e.g.

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
             `here <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`__). 
             Checks will be run for every wildcard in the list. For example ``[\"Glioma*.MRscan.npy\", \"STS*.CTscan.npy\"]``",
        "type": "List[str]"
    }

e.g.

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
        "description": "Path to your data (``MEDscan`` class pickle objects)",
        "type": "str"
    }

e.g.

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

e.g.

.. code-block:: JSON

    {
        "pre_radiomics_checks" : {
            "path_save_checks" : "home/user/medimage/checks",
            }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "path_save_checks",
        "description": "Path where the pre-checks results will be saved",
        "type": "str"
    }

e.g.

.. code-block:: JSON

    {
        "pre_radiomics_checks" : {
            "path_csv" : "home/user/medimage/data/csv/roiNames_GTV.csv",
            }
    }

.. note::
    initializing the :ref:`pre-radiomics checks settings<Pre-checks Parameters>` 
    is optional and can be done in the ``DataManager`` instance initialization step.

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

e.g.

.. code-block:: JSON

    {
        "imParamCT" : {
            "box_string" : "box7",
            }
        "imParamMR" : {
            "box_string" : "box",
            }
        "imParamPET" : {
            "box_string" : "2box",
            }
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

e.g.

.. code-block:: JSON

    {
        "imParamMR" : {
            "interp" : {
                "scale_non_text" : [1, 1, 1],
                "scale_text" : [[1, 1, 1]],
                "vol_interp" : "linear",
                "gl_round" : [],
                "roi_interp" : "linear",
                "roi_pv" : 0.5
            }
        "imParamCT" : {
            "interp" : {
                "scale_non_text" : [2, 2, 3],
                "scale_text" : [[2, 2, 3]],
                "vol_interp" : "nearest",
                "gl_round" : 1,
                "roi_interp" : "nearest",
                "roi_pv" : 0.5
            }
        "imParamPET" : {
            "interp" : {
                "scale_non_text" : [3, 3, 3],
                "scale_text" : [[3, 3, 3]],
                "vol_interp" : "spline",
                "gl_round" : [],
                "roi_interp" : "spline",
                "roi_pv" : 0.5
            }
        }
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

e.g.

.. code-block:: JSON

    {
        "imParamMR" : {
            "reSeg" : {
                "range" : [0, "inf"],
                "outliers" : ""
            }
        },
        "imParamCT" : {
            "reSeg" : {
                "range" : [-500, 500],
                "outliers" : "Collewet"
            }
        },
        "imParamPET" : {
            "reSeg" : {
                "range" : [0, "inf""],
                "outliers" : "Collewet"
            }
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
                            ``\"FBN\"`` for fixed bin number algorithm. Other possible options: ``\"FBSequal\"`` and ``\"FBNequal\"``",
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

e.g. for CT only (the parameters are the same for MR and PET):

.. code-block:: JSON

    {
        "imParamCT" : {
            "discretisation" : {
                "IH" : {
                    "type" : "FBS",
                    "val" : 25
                },
                "IVH" : {
                    "type" : "FBN",
                    "val" : 10
                },
                "texture" : {
                    "type" : ["FBS"],
                    "val" : [[25]]
                }
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "compute_suv_map",
        "description": "Computation of the `suv <https://en.wikipedia.org/wiki/Standardized_uptake_value>`__ map for PET scans. Default ``True``",
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

This parameter is only used for PET scans and is set as follows:

.. code-block:: JSON

    {
        "imParamPET" : {
            "compute_suv_map" : true
            }
    }

.. note::
   This parameter concern PET scans only. ``MEDimage`` only computes suv map for DICOM scans, since the computation relies on 
   DICOM headers for computation and assumes it's already computed for NIfTI scans.

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "filter_type",
        "description": "Name of the filter to use on the scan. Empty string by default.",
        "type": "string",
        "options": {
            "mean": {
                "description": "Filter images using ``mean`` filter.",
                "type": "string"
            },
            "log": {
                "description": "Filter images using ``log`` filter.",
                "type": "string"
            },
            "gabor": {
                "description": "Filter images using ``gabor`` filter.",
                "type": "string"
            },
            "laws": {
                "description": "Filter images using ``laws`` filter.",
                "type": "string"
            },
            "wavelet": {
                "description": "Filter images using ``wavelet`` filter.",
                "type": "string"
            }
        }
    }

e.g.

.. code-block:: JSON

    {
        "imParamPET" : {
            "filter_type" : "mean"
            },
        "imParamMR" : {
            "filter_type" : "laws"
            },
        "imParamCT" : {
            "filter_type" : "log"
            }
    }

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
        "description": "glcm features weighting norm. by default ``False``",
        "title": "glcm dist_correction",
        "type": "Union[bool, str]",
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
                        "description": "Will use discretization length difference corrections as used by the 
                            `Institute of Physics and Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.",
                        "type": "bool"
                    },
                    "False": {
                        "description": "``False`` to replicate IBSI results.",
                        "type": "bool"
                    } 
        }
    }

e.g.

.. code-block:: JSON

    {
        "imParamMR" : {
            "glcm" : {
                "dist_correction" : false
            }
        },
        "imParamCT" : {
            "glcm" : {
                "dist_correction" : "chebyshev"
            }
        },
        "imParamPET" : {
            "glcm" : {
                "dist_correction" : "euclidean"
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "glcm features aggregation method. by default ``\"vol_merge\"``",
        "title": "glcm merge_method",
        "type": "string",
        "options": {
                    "vol_merge": {
                        "description": "Features are extracted from a single matrix after merging all 3D directional matrices.",
                        "type": "string"
                    },
                    "slice_merge": {
                        "description": "Features are extracted from a single matrix after merging 2D directional matrices per slice,
                            and then averaged over slices.",
                        "type": "string"
                    },
                    "dir_merge": {
                        "description": "Features are extracted from a single matrix after merging 2D directional matrices per direction, 
                            and then averaged over direction",
                        "type": "string"
                    },
                    "average": {
                        "description": "Features are extracted from each 3D directional matrix and averaged over the 3D directions",
                        "type": "string"
                    }
        }
    }

e.g.

.. code-block:: JSON

    {
        "imParamMR" : {
            "glcm" : {
                "merge_method" : "average"
            }
        },
        "imParamCT" : {
            "glcm" : {
                "merge_method" : "vol_merge"
            }
        },
        "imParamPET" : {
            "glcm" : {
                "merge_method" : "dir_merge"
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "glrlm features weighting norm. by default ``False``",
        "title": "glrlm dist_correction",
        "type": "Union[bool, str]",
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
                        "description": "Will use discretization length difference corrections as used by the 
                            `Institute of Physics and Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.",
                        "type": "bool"
                    },
                    "False": {
                        "description": "``False`` to replicate IBSI results.",
                        "type": "bool"
                    } 
        }
    }

e.g.

.. code-block:: JSON

    {
        "imParamMR" : {
            "glrlm" : {
                "dist_correction" : false
            }
        },
        "imParamCT" : {
            "glrlm" : {
                "dist_correction" : "chebyshev"
            }
        },
        "imParamPET" : {
            "glrlm" : {
                "dist_correction" : "euclidean"
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "glrlm features aggregation method. by default ``\"vol_merge\"``",
        "title": "glrlm merge_method",
        "type": "string",
        "options": {
                    "vol_merge": {
                        "description": "Features are extracted from a single matrix after merging all 3D directional matrices.",
                        "type": "string"
                    },
                    "slice_merge": {
                        "description": "Features are extracted from a single matrix after merging 2D directional matrices per slice,
                            and then averaged over slices.",
                        "type": "string"
                    },
                    "dir_merge": {
                        "description": "Features are extracted from a single matrix after merging 2D directional matrices per direction, 
                            and then averaged over direction",
                        "type": "string"
                    },
                    "average": {
                        "description": "Features are extracted from each 3D directional matrix and averaged over the 3D directions",
                        "type": "string"
                    }
        }
    }

e.g.

.. code-block:: JSON

    {
        "imParamMR" : {
            "glrlm" : {
                "merge_method" : "average"
            }
        },
        "imParamCT" : {
            "glrlm" : {
                "merge_method" : "vol_merge"
            }
        },
        "imParamPET" : {
            "glrlm" : {
                "merge_method" : "dir_merge"
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "ngtdm features weighting norm. by default ``False``",
        "title": "ngtdm dist_correction",
        "type": "bool",
        "options": {
                    "True": {
                        "description": "Will use discretization length difference corrections as used by the 
                            `Institute of Physics and Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.",
                        "type": "bool"
                    },
                    "False": {
                        "description": "``False`` to replicate IBSI results.",
                        "type": "bool"
                    }
        }
    }

e.g.

.. code-block:: JSON

    {
        "imParamMR" : {
            "ngtdm" : {
                "dist_correction" : false
            }
        },
        "imParamCT" : {
            "ngtdm" : {
                "dist_correction" : true
            }
        },
        "imParamPET" : {
            "ngtdm" : {
                "dist_correction" : true
            }
        }
    }


Filtering parameters
--------------------

Filtering parameters are organized  in a separate dictionary, each dictionary contains 
parameters for every filter of the ``MEDimage``:

.. code-block:: JSON

    {
        "imParamFilter": {
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

e.g.

.. code-block:: JSON

    {
        "imParamFilter" : {
            "mean" : {
                "ndims" : 3,
                "size" : 5,
                "padding" : "symmetric",
                "name_save" : "mean5"
            }
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

e.g.

.. code-block:: JSON

    {
        "imParamFilter" : {
            "log" : {
                "ndims" : 3,
                "sigma" : 1.5,
                "orthogonal_rot" : false,
                "padding" : "constant",
                "name_save" : "log_1.5"
            }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "laws",
        "description": "Parameters of the laws filter",
        "type": "dict",
        "options": {
            "config": {
                "description": "List of string of every 1D filter to use for the Laws kernel creation. Possible 1D filters:
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

e.g.

.. code-block:: JSON

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
            }
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
                    maps of several elements of the Gabor filter bank.",
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

e.g.

.. code-block:: JSON

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
            }
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
                "description": "Wavelet name used to create the kernel. The Wavelet families and built-ins can be 
                    found `here <https://pywavelets.readthedocs.io/en/v0.3.0/ref/wavelets.html#wavelet-families>`__.
                    Custom user wavelets are also supported.",
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

e.g.

.. code-block:: JSON

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
    }

Example of a full settings dictionary
-------------------------------------

Here is an example of a complete settings dictionary:

.. raw:: html

    <div id="json_dict"></div>

    <script>
    $(document).ready(function () {
        var json_dict = {
            "pre_radiomics_checks" : {
                "path_data" : "",
                "wildcards_dimensions" : [
                "Glioma*.MRscan.npy"
                ],
                "path_csv" : "",
                "wildcards_window" : [
                "Glioma*.MRscan.npy"
                ],
                "path_save_checks" : ""
            },
            "n_batch" : 16,
            "roi_type_labels" : [
                "Lesions"
            ],
            "roi_types" : [
                "CTLesion"
            ],
            "imParamMR" : {
                "box_string": "box10",
                "interp" : {
                "scale_non_text" : [2, 2, 3],
                "scale_text" : [[2, 2, 3]],
                "vol_interp" : "linear",
                "gl_round" : 1,
                "roi_interp" : "linear",
                "roi_pv" : 0.5
                },
                "reSeg" : {
                "range" : [-500, "inf"],
                "outliers" : ""
                },
                "discretisation" : {
                "IH" : {
                    "type" : "FBS",
                    "val" : 25
                },
                "IVH" : {

                },
                "texture" : {
                    "type" : ["FBS"],
                    "val" : [[25]]
                }
                },
                "glcm" : {
                    "dist_correction" : "Chebyshev",
                    "merge_method": "vol_merge"
                },
                "glrlm" : {
                    "dist_correction" : false,
                    "merge_method": "vol_merge"
                },
                "ngtdm" : {
                    "dist_correction" : false
                },
                "filter_type": ""
                },
            "imParamCT" : {
                "interp" : {
                "scale_non_text" : [2, 2, 2],
                "scale_text" : [[2, 2, 2]],
                "vol_interp" : "linear",
                "gl_round" : 1,
                "roi_interp" : "linear",
                "roi_pv" : 0.5
                },
                "reSeg" : {
                "range" : [-1000,400],
                "outliers" : ""
                },
                "discretisation" : {
                "IH" : {
                    "type" : "FBS",
                    "val" : 25
                },
                "IVH" : {
                    "type" : "FBS",
                    "val" : 2.5
                },
                "texture" : {
                    "type" : ["FBS"],
                    "val" : [[25]]
                }
                },
                "glcm" : {
                    "dist_correction" : false,
                    "merge_method": "vol_merge"
                },
                "glrlm" : {
                    "dist_correction" : false,
                    "merge_method": "vol_merge"
                },
                "ngtdm" : {
                    "dist_correction" : false
                },
                "filter_type": ""
                },
            "imParamPET" : {
                "compute_suv_map" : true,
                "interp" :  {
                    "scale_non_text" : [4, 4, 4],
                    "scale_text" : [[3, 3, 3], [4, 4, 4]],
                    "vol_interp" : "linear",
                    "gl_round" : [],
                    "roi_interp" : "linear",
                    "roi_pv" : 0.5
                },
                "reSeg" :  {
                    "range" : [0, "inf"],
                    "outliers" : ""
                },
                "discretisation" :  {
                    "IH" : {
                    "type" : "FBN",
                    "val" : 64
                    },
                    "IVH" : {
                    "type" : "FBS",
                    "val" : 0.1
                    },
                    "texture" : {
                    "type" : ["FBS", "FBSequal"],
                    "val" : [[0.5, 1], [0.5, 1]]
                    }
                },
                "glcm" : {
                    "dist_correction" : "Chebyshev",
                    "merge_method": "vol_merge"
                },
                "glrlm" : {
                    "dist_correction" : false,
                    "merge_method": "vol_merge"
                },
                "ngtdm" : {
                    "dist_correction" : false
                },
                "filter_type": ""
            },
            "imParamFilter" : {
                "mean" : {
                "ndims" : 3,
                "size" : 5,
                "padding" : "symmetric",
                "orthogonal_rot" : false,
                "name_save" : ""
                },
                "log" : {
                "ndims" : 3,
                "sigma" : 1.5,
                "orthogonal_rot" : false,
                "padding" : "symmetric",
                "name_save" : ""
                },
                "laws" : {
                "config" : ["L5", "E5", "E5"],
                "energy_distance" : 7,
                "rot_invariance" : true,
                "orthogonal_rot" : false,
                "energy_image" : true,
                "padding" : "symmetric",
                "name_save" : ""
                },
                "gabor" : {
                "sigma" : 5,
                "lambda" : 2,
                "gamma" : 1.5,
                "theta" : "Pi/8",
                "rot_invariance" : true,
                "orthogonal_rot" : true,
                "padding" : "symmetric",
                "name_save" : ""
                },
                "wavelet" : {
                "ndims" : 3,
                "basis_function" : "db3",
                "subband" : "LLH",
                "level" : 1,
                "rot_invariance" : true,
                "padding" : "symmetric",
                "name_save" : "Wavelet_db3_LLH"
                }
            }
            }
            ;
        $("#json_dict").html(
            "<a href='#' class='json_dict_expand'>Click here to display the dictionary</a>" +
            "<pre class='json_dict_content' style='display: none;'>" +
            JSON.stringify(json_dict, null, 4) +
            "</pre>"
        );
        $(".json_dict_expand").click(function (e) {
            e.preventDefault();
            $(".json_dict_content").toggle();
        });
    });
    </script>



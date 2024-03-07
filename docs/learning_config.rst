Learning
--------

This section will walk you through the details on how to set up the configuration file for the machine learning part of the pipeline. 
It will be separated to the following subdivisions:

- :ref:`Design<Experiment Design Parameters>`
- :ref:`Data Cleaning<Data Cleaning Parameters>`
- :ref:`Data Normalization<Data Normalization Parameters>`
- :ref:`Feature Set Reduction<Feature Set Reduction Parameters>`
- :ref:`Machine Learning<Machine Learning Parameters>`
- :ref:`Variables Definition<Variables Definition>`

Experiment Design Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This set of parameters is used to define the experiment design (data splitting, splitting proportion...), it is organized as follows:

.. code-block:: JSON

    {
        "testSets": ["Define method here"],
        "method name": "Define method here"
        
    }

Now let's specify the parameters for the selected method; for instance, in the case of the ``Random`` and ``CV`` methods:

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Splitting methods",
        "description": "Type of sets to create.",
        "type": "object",
        "properties": {
            "Random": {
                "description": "Random splitting method.",
                "type": "object",
                "properties": {
                    "method": {
                        "description": "Method of splitting the data.",
                        "type": "string",
                        "options": {
                            "SubSampling": {
                                "description": "The data will be randomly split",
                                "type": "string"
                            },
                            "Institutions": {
                                "description": "The data will be split based on institutions",
                                "type": "string"
                            }
                        }
                    },
                    "nSplits": {
                        "description": "Number of splits to create.",
                        "type": "int"
                    },
                    "stratifyInstitutions": {
                        "description": "If ``True``, the data will be stratified based on institutions.",
                        "type": "bool"
                    },
                    "testProportion": {
                        "description": "Proportion of the test set.",
                        "type": "float"
                    },
                    "seed": {
                        "description": "Seed for the random number generator.",
                        "type": "int"
                    }
                }
            },
            "CV" : {
                "description": "Cross-validation splitting method.",
                "type": "object",
                "properties": {
                    "nFolds": {
                        "description": "Number of folds to use.",
                        "type": "int"
                    },
                    "seed": {
                        "description": "Seed for the random number generator.",
                        "type": "int"
                    }
                }
            }
        }
    }

- **Example**

.. code-block:: JSON

    {
        "Random": {
            "method": "SubSampling",
            "nSplits": 10,
            "stratifyInstitutions": 1,
            "testProportion": 0.33,
            "seed": 54288
        }
    }


Data Cleaning Parameters
^^^^^^^^^^^^^^^^^^^^^^^^
This set of parameters is used to define the data cleaning process Parameters, it is organized as follows:

.. code-block:: JSON

    {
	    "method name": {
            "define parameters here"
        },
        "another method": {
            "define parameters here"
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Cleaning methods",
        "description": "Feature cleaning method name.",
        "type": "object",
        "properties": {
            "default": {
                "description": "Default cleaning method.",
                "type": "string"
            }
        }
    }

Now let's specify the parameters for the selected cleaning method; for instance, in the case of the ``default`` method:

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Chosen method's parameters",
        "description": "Feature cleaning parameters.",
        "type": "object",
        "properties": {
            "continuous": {
                "description": "Continuous feature cleaning parameters.",
                "type": "object",
                "properties": {
                    "missingCutoffps": {
                        "description": "Maximum percentage cut-offs of missing features per sample. Samples with more missing features than this cut-off will be removed.",
                        "type": "float"
                    },
                    "covCutoff": {
                        "description": "Minimal coefficient of variation cut-offs over samples per variable. Variables with less coefficient of variation than this cut-off will be removed.",
                        "type": "float"
                    },
                    "missingCutoffpf": {
                        "description": "Maximal percentage cut-offs of missing samples per variable. Features with more missing samples than this cut-off will be removed.",
                        "type": "float"
                    },
                    "imputation": {
                        "description": "Imputation method for missing values. Default is ``mean``.",
                        "type": "string",
                        "options": {
                            "mean": {
                                "description": "Impute missing values with the mean of the feature.",
                                "type": "string"
                            },
                            "median": {
                                "description": "Impute missing values with the median of the feature.",
                                "type": "string"
                            },
                            "random": {
                                "description": "Impute missing values with the a random value from the feature set.",
                                "type": "string"
                            }
                        }
                    }
                }
            }
        }
    }

- **Example**

.. code-block:: JSON

    {
        "default": 
        {
        "feature": {
			"continuous": {
				"missingCutoffps": 0.25,
				"covCutoff": 0.1,
				"missingCutoffpf": 0.1,
				"imputation": "mean"
            }
        }
    }

.. note::
    Note that you can add as many methods as you want, for other feature types (categorical, ordinal, etc.) and for other cleaning methods (e.g. ``PCA``).

Data Normalization Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data normalization aims to remove batch effects from the data. This set of parameters is used to define the data normalization process Parameters, it is organized as follows:

.. code-block:: JSON

    {
        "standardCombat": {
            "define parameters here"
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Chosen method parameters",
        "description": "Normalization method name.",
        "type": "string",
        "options": {
            "standardCombat": {
                "description": "Standard Combat normalization method.",
                "type": "string"
            }
        }
    }

.. note::
    For now only the ``standardCombat`` method is available and it does not require any parameters.

Feature Set Reduction Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Feature set reduction consists of reducing the number of features in the data by removing correlated features, selecting important features, etc. This set of parameters is used to define the feature set reduction process Parameters, it is organized as follows:

.. code-block:: JSON

    {
        "selected method": {
            "define parameters here"
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "method name",
        "description": "Feature set reduction method name.",
        "type": "string",
        "options": {
            "FDA": {
                "description": "False discovery avoidance method. `Read the paper. <https://ieeexplore.ieee.org/document/8528467>`__",
                "type": "string"
            },
            "FDAbalanced": {
                "description": "Balanced version of the False discovery avoidance method, where the selected number of features is the same for each table.",
                "type": "string"
            }
        }
    }

Now let's specify the parameters for the selected feature set reduction method; for instance, in the case of the ``FDA`` method:

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "FDA method",
        "description": "Feature set reduction parameters.",
        "type": "object",
        "properties": {
            "FDA": {
                "description": "FDA method's parameters.",
                "type": "object",
                "properties": {
                    "nSplits": {
                        "description": "Number of splits to use for the FDA algorithm.",
                        "type": "int"
                    },
                    "corrType": {
                        "description": "Type of correlation to use for the FDA algorithm. Default is ``Spearman``.",
                        "type": "string",
                        "options": {
                            "Spearman": {
                                "description": "Spearman correlation.",
                                "type": "string"
                            },
                            "Pearson": {
                                "description": "Pearson correlation.",
                                "type": "string"
                            }
                        }
                    },
                    "threshStableStart": {
                        "description": "Stability threshold to cut-off the unstable features at the beginning of the FDA algorithm.",
                        "type": "float"
                    },
                    "threshInterCorr": {
                        "description": "Threshold to cut-off the inter-correlated features.",
                        "type": "float"
                    },
                    "minNfeatStable": {
                        "description": "Minimum number of stable features to keep before inter-correlation step.",
                        "type": "int"
                    },
                    "minNfeatInterCorr": {
                        "description": "Minimum number of inter-correlated features to keep.",
                        "type": "int"
                    },
                    "minNfeat": {
                        "description": "Minimum number of features to keep at the end of the FDA algorithm.",
                        "type": "int"
                    },
                    "seed": {
                        "description": "Seed for the random number generator.",
                        "type": "int"
                    }
                }
            }
        }
    }

- **Example**

.. code-block:: JSON

    {
        "FDA": {
            "nSplits": 100,
            "corrType": "Spearman",
            "threshStableStart": 0.5,
            "threshInterCorr": 0.7,
            "minNfeatStable": 100,
            "minNfeatInterCorr": 60,
            "minNfeat": 5,
            "seed": 54288
        }
    }

.. note::
    Only ``FDA`` and ``FDAbalanced`` methods are available for now and they share the same parameters.

Machine Learning Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This set of parameters is used to define the machine learning process, algorithm, and parameters, it is organized as follows:

.. code-block:: JSON

    {
        "selected algorithm": {
            "define parameters here"
        }
    }

Now let's specify the parameters for the selected machine learning algorithm; for instance, in the case of the ``XGBoost`` algorithm:

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "ML Algorithm",
        "description": "Machine learning algorithm name.",
        "type": "object",
        "properties": {
            "XGBoost": {
                "description": "`XGBoost <https://xgboost.readthedocs.io/en/latest/>`__ algorithm.",
                "type": "object",
                "properties": {
                    "varImportanceThreshold": {
                        "description": "Variable importance threshold. Default is ``0.3``. Variables with importance below this threshold will be removed.",
                        "type": "float"
                    },
                    "optimalThreshold": {
                        "description": "If ``null``, the optimal threshold will be computed. Default is ``0.5``.",
                        "type": "float"
                    },
                    "optimizationMetric": {
                        "description": "Model's optimization metric. Default is ``AUC``. Only used if ``method`` is ``pycaret``.",
                        "type": "string"
                    },
                    "method": {
                        "description": "Method to use for the XGBoost algorithm. Default is ``pycaret``.",
                        "type": "string",
                        "options": {
                            "pycaret": {
                                "description": "Automated using `PyCaret <https://pycaret.org/>`__.",
                                "type": "string"
                            },
                            "random_search": {
                                "description": "Random search using a pre-defined grid of parameters.",
                                "type": "string"
                            },
                            "grid_search": {
                                "description": "Grid search using a pre-defined grid of parameters.",
                                "type": "string"
                            }
                        }
                    },
                    "nameSave" : {
                        "description": "Name of the file to save the model.",
                        "type": "string"
                    },
                    "seed" : {
                        "description": "Seed for the random number generator.",
                        "type": "int"
                    }
                }
            }
        }
    }

- **Example**

.. code-block:: JSON

    {
        "XGBoost": {
            "varImportanceThreshold": 0.3,
            "optimalThreshold": null,
            "optimizationMetric": "AUC",
            "method": "pycaret",
            "nameSave": "XGBoost03AUC",
            "seed": 54288
        }
    }

.. note::
    Only the ``XGBoost`` algorithm is available for now.

Variables Definition
^^^^^^^^^^^^^^^^^^^^

This set of parameters is used to define the variables to use for the machine learning process, it is organized as follows:

.. code-block:: JSON

    {
        "selected variable": {
            "define parameters here"
        },
        "combinations": [
            "Insert combinations of variables here"
        ]
    }

.. jsonschema::
    
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "title": "Variables",
            "description": "Variables to use for the machine learning process.",
            "type": "object",
            "properties": {
                "combinations": {
                    "description": "List of variables combinations to use for the study.",
                    "type": "List[str]"
                }
            }
        }

For the selected variable, you can specify the following parameters:

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "selected variable",
        "description": "Variable name to use for the machine learning process.",
        "type": "object",
        "properties": {
            "nameType": {
                "description": "Type of variable to use. Must contain ``Radiomics`` for radiomics features.",
                "type": "string"
            },
            "path": {
                "description": "Path to the variable file. Use ``\"setToFolderNameinWorkspace\"`` to set the features folder to ``FolderName`` in the workspace.",
                "type": "string"
            },
            "scans": {
                "description": "List of scans to use for the variable. For example is ``T1C``.",
                "type": "List[str]"
            },
            "rois": {
                "description": "List of ROIs to include in the study (will be used to identify the features fie). For example is ``GTV``.",
                "type": "List[str]"
            },
            "imSpaces": {
                "description": "Radiomics level, the features file must end with this level. For example is ``morph``.",
                "type": "List[str]"
            },
            "var_datacleaning": {
                "description": "Data cleaning method to use for the variable. Default is ``default``.",
                "type": "string"
            },
            "var_normalization": {
                "description": "Data normalization method to use for the variable. Default is ``combat``.",
                "type": "string"
            },
            "var_fSetReduction": {
                "description": "Feature set reduction method to use for the variable. Default is ``FDA``.",
                "type": "string"
            }
        }
    }

- **Example**

.. code-block:: JSON

    {
        "var1": {
            "nameType": "RadiomicsMorph",
            "path": "setToMyFeaturesInWorkspace",
            "scans": ["T1CE"],
            "rois": ["GTV"],
            "imSpaces": ["morph"],
            "var_datacleaning": "default",
            "var_normalization": "combat",
            "var_fSetReduction": "FDA"
        },
        "combinations": [
            "var1"
        ]
    }

<div align="center">

<img src="https://raw.githubusercontent.com/MahdiAll99/MEDimage/dev/docs/figures/MEDimageLogo.png" style="width:150px;"/>

[![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-380/)
[![TestPyPI - version](https://img.shields.io/badge/pypi-v0.2.0-blue)](https://test.pypi.org/project/medimage-pkg/0.2.0/)
[![Continuous Integration](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-app.yml/badge.svg)](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-app.yml)
[![Upload Python Package](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-publish.yml/badge.svg)](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-publish.yml)
[![Documentation Status](https://readthedocs.org/projects/medimage/badge/?version=latest)](https://medimage.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/DataManager-Tutorial.ipynb)

</div>

## Table of Contents
  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. Generating the documentation locally](#3-generating-the-documentation-locally)
  * [5. Tutorials](#5-tutorials)
  * [6. IBSI Standardization](#6-ibsi-standardization)
    * [IBSI Chapter 1](#ibsi-chapter-1)
    * [IBSI Chapter 2 (In progress)](#ibsi-chapter-2-in-progress)
  * [7. Acknowledgement](#7-acknowledgement)
  * [8. Authors](#8-authors)
  * [9. Statement](#9-statement)

## 1. Introduction
This is an open-source python package for processing and features extracting from multi-modal (MRI, CT and PET) medical images. It facilitates the medical-images processing and the computation of all types of radiomic features as well as the reproducibility of the different analysis. This package has been standardized with the [IBSI](https://theibsi.github.io/) norms.


![MEDimage overview](https://raw.githubusercontent.com/MahdiAll99/MEDimage/main/docs/figures/pakcage-overview.png)


## 2. Installation

### Python installation
The MEDimage package requires *Python 3.8* or more. If you don't have it installed  on your machine, follow the instructions [here](https://github.com/MahdiAll99/MEDimage/blob/main/python.md) to install it.

### Package installation
You can easily install the ``MEDimage``package from PyPI using:
```
pip install MEDimage
```

For more installation options (conda, poetry...) check out the [installation documentation](https://medimage.readthedocs.io/en/latest/Installation.html).

## 3. Generating the documentation locally
We used sphinx to create the documentation for this project and you can check it out in this [link](https://medimage.readthedocs.io/en/latest/). However you can generate and host it locally by compiling the documentation source code using:

```
cd docs
make clean
make html
```

Then open it locally using:

```
cd _build/html
python -m http.server
```

## 4. A simple example
```python
import os
import pickle

import MEDimage

# Load the DataManager
dm = MEDimage.DataManager(path_dicoms=os.getcwd())

# Process the DICOM files and retrieve the MEDimage object
med_obj = dm.process_dicoms()

# Extract ROI mask from the object
vol_obj_init, roi_obj_init = MEDimage.processing.get_roi_from_indexes(
            med_obj,
            name_roi='{ED}+{ET}+{NET}',
            box_string='full')

# Extract features from the imaging data
local_intensity = MEDimage.biomarkers.local_intensity.extract_all(
                img_obj=vol_obj_init.data,
                roi_obj=roi_obj_init.data,
                res=[1, 1, 1]
            )

# Update radiomics results class
med_obj.update_radiomics(loc_int_features=local_intensity)

# Saving radiomics results
med_obj.save_radiomics(
                scan_file_name='STS-UdS-001__T1.MRscan.npy',
                path_save=os.getcwd(),
                roi_type='GrossTumorVolume',
                roi_type_label='GTV',
            )
```

## 5. Tutorials

We have created many [tutorial notebooks](https://github.com/MahdiAll99/MEDimage/tree/main/notebooks) to assist you in learning how to use the different parts of the package. More details can be found in the [documentation](https://medimage.readthedocs.io/en/latest/tutorials.html).

## 6. IBSI Standardization
The image biomarker standardization initiative ([IBSI](https://theibsi.github.io)) is an independent international collaboration that aims to standardize the extraction of image biomarkers from acquired imaging. The IBSI therefore seeks to provide image biomarker nomenclature and definitions, benchmark datasets, and benchmark values to verify image processing and image biomarker calculations, as well as reporting guidelines, for high-throughput image analysis. We participate in this collaboration with our package to make sure it respects the international nomenclatures and definitions. The participation was separated to two chapters:

  - ### IBSI Chapter 1
      [The IBSI chapter 1](https://theibsi.github.io/ibsi1/) is dedicated to the standardization of commonly used radiomic features. It was initiated in September 2016 and reached completion in March 2020. We have created two [jupyter notebooks](https://github.com/MahdiAll99/MEDimage/tree/main/notebooks/ibsi) for each phase of the chapter and made them available for the users to run the IBSI tests for themselves. The tests can also be explored in interactive Colab notebooks that are directly accessible here:
      
      - **Phase 1**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi1p1.ipynb)
      - **Phase 2**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi1p2.ipynb)

  - ### IBSI Chapter 2 (In progress)
      [The IBSI chapter 2](https://theibsi.github.io/ibsi2/) was launched in June 2020 and still in progress. It is dedicated to the standardization of commonly used imaging filters in radiomic studies. We have created two [jupyter notebooks](https://github.com/MahdiAll99/MEDimage/tree/main/notebooks/ibsi) for each phase of the chapter and made them available for the users to run the IBSI tests for themselves and validate image filtering and image biomarker calculations from filter response maps. The tests can also be explored in interactive Colab notebooks that are directly accessible here: 
      
      - **Phase 1**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi2p1.ipynb)
      - **Phase 2**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi2p2.ipynb)

      Our team *UdeS* (a.k.a. Université de Sherbrooke) has already submitted the benchmarked values to the IBSI uploading website and it can be found here: [IBSI upload page](https://ibsi.radiomics.hevs.ch/).

---
**Miscellaneous**

You can avoid the next steps (jupyter installation and environment set up) if you installed the package using conda or poetry according to the documentation.

---

You can view & run the tests locally by installing the [Jupyter Notebook](https://jupyter.org/) application on your machine:
```
python -m pip install jupyter
```
Then add the installed `medimage` environment to the jupyter notebook kernels using:

```
python -m ipykernel install --user --name=medimage
```

Then access the IBSI tests folder using:

```
cd notebooks/ibsi/
```

Finally, launch jupyter notebook to navigate through the IBSI notebooks using:

```
jupyter notebook
```

## 7. Acknowledgement
MEDimage is an open source package developed at the [MEDomics-Udes](https://www.medomics-udes.org/en/) laboratory with the collaboration of the international consortium [MEDomics](https://www.medomics.ai/). We welcome any contribution and feedback. Furthermore, we wish that this package could serve the growing radiomics research community by providing a flexible as well as [IBSI](https://theibsi.github.io/) standardized tool to reimplement existing methods and develop their own new methods.

## 8. Authors
* [MEDomics](https://github.com/medomics/): MEDomics consortium.

## 9. Statement

This package is part of https://github.com/medomics, a package providing research utility tools for developing precision medicine applications.

```
MIT License

Copyright (C) 2022 MEDomics consortium

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

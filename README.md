[![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/medimage/badge/?version=latest)](https://medimage.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-GPL%203.0-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/MEDimage-Tutorial.ipynb)


# MEDimage

## Table of Contents
  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. Generating the Documentation Locally](#3-generating-the-documentation-locally)
  * [4. IBSI Tests](#4-ibsi-tests)
    * [IBSI Chapter 1](#ibsi-chapter-1)
    * [IBSI Chapter 2](#ibsi-chapter-2)
  * [5. Project Files Organization](#5-project-files-organization)
  * [6. Authors](#6-authors)
  * [7. Statement](#7-statement)

## 1. Introduction
*MEDimage* is a Python package for processing and extracting features from medical images. It gives you the ability to process and filter images and compute all types of radiomic features. This package has been standardized with the [IBSI](https://theibsi.github.io/) norms.

## 2. Installation

### Python installation
The MEDimage package requires *python 3.8* or more to be run. If you don't have it installed  on your machine, follow the instructions [here](https://github.com/MahdiAll99/MEDimage/blob/main/python.md).

### Cloning the repository
In your terminal, clone the repository
```
git clone https://github.com/MahdiAll99/MEDimage.git
```

Then access the package directory using:
```
cd MEDimage
```

### Making the environment
In order to use the package, make sure to have Anaconda distribution on your machine, you can download and install it by following the instructions on [this link](https://docs.anaconda.com/anaconda/install/index.html).

Using anaconda distribution, we will create and activate the `medimage` environment. You can do so by running this command

```
make -f Makefile.mk create_environment
```

This command will install all the dependencies required. And now we  initialize conda with

```
conda init
```

And activate the `medimage` environment  using

```
conda activate medimage
```

Once the environment is activated, you can generate the documentation in the [doc section](#3-generating-the-documentation-locally) or start running the [IBSI-Tests](#4-ibsi-tests) without documentation (not recommended).

## 3. Generating the Documentation Locally
The package documentation can be generated locally using [pdoc3](https://pdoc.dev/docs/pdoc.html).

From your terminal, from the MEDimage package folder use the following command to generate the documentation
```
pdoc3 --http localhost:8080 -c latex_math=True MEDimage
```

The documentation will be available on the *8080 localhost* via the link http://localhost:8080/MEDimage/. The IBSI tests can now be run by following the [IBSI Tests](#4-ibsi-tests) section.

## 4. IBSI Tests
The image biomarker standardization initiative (IBSI) is an independent international collaboration which works towards standardizing the extraction of image biomarkers from acquired imaging. The IBSI therefore seeks to provide image biomarker nomenclature and definitions, benchmark data sets, and benchmark values to verify image processing and image biomarker calculations, as well as reporting guidelines, for high-throughput image analysis.

  - ### IBSI Chapter 1
      [The IBSI chapter 1](https://theibsi.github.io/ibsi1/) was initiated in September 2016, and it reached completion in March 2020 and is dedicated to the standardization of commonly used radiomic features. Notebooks are available to test and understand the MEDimage package implementations and validate image processing and image biomarker calculations.

  - ### IBSI Chapter 2
      [The IBSI chapter 2](https://theibsi.github.io/ibsi2/) was launched in June 2020 and still ongoing. It is dedicated to the standardization of commonly used imaging filters in radiomic studies. Notebooks are available to test and understand the MEDimage package implementations and validate image filtering and image biomarker calculations from filter response maps. Our team *UdeS* (a.k.a. Université de Sherbrooke) has already submitted the benchmarked values to the IBSI uploading website and it can be found here: [IBSI upload page](https://ibsi.radiomics.hevs.ch/).

First, we need to install the [Jupyter Notebook](https://jupyter.org/) application on our machine by running
```
python -m pip install jupyter
```
Second, before we run the notebooks we need to add the installed `medimage` environment to the jupyter notebook kernels using

```
python -m ipykernel install --user --name=medimage
```

Then access the IBSI tests folder using

```
cd IBSI-TESTs
```

Finally, we launch jupyter notebook to navigate through the IBSI notebooks and have fun testing

```
jupyter notebook
```

## 5. Project Files Organization
```
├── LICENSE
├── Makefile           <- Makefile with multiple commands for the environment setup.
├── README.md          <- The main README with Markdown language for this package.
├── IBSI-TESTs
│   ├── data           <- Data from IBSI.
│   ├── images         <- Figures used in the notebooks.
│   ├── settings       <- JSON files for configurations for each IBSI test.
│   ├── ibsi1p1.ipynb  <- IBSI chapter 1 phase 1 tutorial.
│   ├── ibsi1p2.ipynb  <- IBSI chapter 1 phase 2 tutorial.
│   ├── ibsi2p1.ipynb  <- IBSI chapter 2 phase 1 tutorial.
│   └── ibsi2p2.ipynb  <- IBSI chapter 2 phase 2 tutorial.
|
├── MEDimage           <- Source code for the MEDimage mother class and all the child classes
|   |                     and for image filtering as well.
│   ├── utils          <- Scripts for all the useful methods that will be called in other parts
|   |                     of the package.
│   ├── processing     <- Scripts for image processing (interpolation, re-segmentation...).
│   ├── biomarkers     <- Scripts to extract features and do all features-based computations.
│   ├── __init__.py    <- Makes MEDimage a Python module.
│   ├── Filter.py
│   ├── MEDimage.py
│   ├── MEDimageProcessing.py
│   └── MEDimageComputeRadiomics.py
|
├── environment.py     <- The dependencies file to create the `medimage` environment.
│
└── setup.py           <- Allows MEDimage package to be installed as a python package.
```

## 6. Authors
* [MEDomics](https://github.com/medomics/): MEDomics consortium

## 7. Statement

This package is part of https://github.com/medomics, a package providing research utility tools for developing precision medicine applications.

--> Copyright (C) 2022 MEDomics consortium

```
This package is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this package.  If not, see <http://www.gnu.org/licenses/>.
```

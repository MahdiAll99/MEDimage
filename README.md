<div align="center">

<img src="docs/figures/MEDimageLogo.png" width="200"/>

[![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/medimage/badge/?version=latest)](https://medimage.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/tutorial/DataManager-Tutorial.ipynb)

</div>

## Table of Contents
  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. Generating the documentation locally](#3-generating-the-documentation-locally)
  * [4. IBSI Tests](#4-ibsi-tests)
    * [IBSI Chapter 1](#ibsi-chapter-1)
    * [IBSI Chapter 2 (In progress)](#ibsi-chapter-2-in-progress)
  * [5. Project Files Organization](#5-project-files-organization)
  * [6. Authors](#6-authors)
  * [7. Statement](#7-statement)

## 1. Introduction
*MEDimage* is an open-source python package for processing and extracting features from medical images. It facilitates the medical-images processing and computation of all types of radiomic features as well as the reproducibility of the different analysis. This package has been standardized with the [IBSI](https://theibsi.github.io/) norms.

![MEDimage overview](https://github.com/MahdiAll99/MEDimage/blob/user-doc/docs/figures/pakcage-overview.png?raw=true)

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

## 3. Generating the documentation locally
We used sphinx to create the documentation for this project and it can be found in the following [link](https://medimage.readthedocs.io/en/latest/). But you can generate and host it locally by compiling the documentation source code using:

```
make clean
make html
```
Then open it locally using:
```
cd _build/html
python -m http.server
```

## 4. Tutorials

We have created many tutorials for the different functionalities of the package. More details can be found in the [documentation](https://medimage.readthedocs.io/en/user-doc/tutorials.html)

## 4. IBSI Tests
The image biomarker standardization initiative (IBSI) is an independent international collaboration which works towards standardizing the extraction of image biomarkers from acquired imaging. The IBSI therefore seeks to provide image biomarker nomenclature and definitions, benchmark data sets, and benchmark values to verify image processing and image biomarker calculations, as well as reporting guidelines, for high-throughput image analysis. We have participated in this collaboration with our package to make sure it respects the international nomenclatures and definitions The participation was separated to two chapters:

  - ### IBSI Chapter 1
      [The IBSI chapter 1](https://theibsi.github.io/ibsi1/) was initiated in September 2016, and it reached completion in March 2020 and is dedicated to the standardization of commonly used radiomic features. We have created jupyter notebooks and made it available for the users to run the IBSI tests themselves. The tests are an interactive Colab notebooks and is directly accessible here: 
      
      - **Phase 1**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/ibsi/ibsi1p1.ipynb)
      - **Phase 2**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/ibsi/ibsi1p2.ipynb)

  - ### IBSI Chapter 2 (In progress)
      [The IBSI chapter 2](https://theibsi.github.io/ibsi2/) was launched in June 2020 and still in progress. It is dedicated to the standardization of commonly used imaging filters in radiomic studies. We have created jupyter notebooks and made it available for the users to run the IBSI tests themselves and validate image filtering and image biomarker calculations from filter response maps. The tests are an interactive Colab notebooks and is directly accessible here: 
      
      - **Phase 1**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/ibsi/ibsi2p1.ipynb)
      - **Phase 2**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/dev/notebooks/ibsi/ibsi2p2.ipynb)

      Our team *UdeS* (a.k.a. Université de Sherbrooke) has already submitted the benchmarked values to the IBSI uploading website and it can be found here: [IBSI upload page](https://ibsi.radiomics.hevs.ch/).

You can also view & run the tests locall by installing the [Jupyter Notebook](https://jupyter.org/) application on your machine:
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

```
MIT License

Copyright (C) 2022 MEDomics consortium

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

Installation
============

Python installation
-------------------

The MEDimage package requires python 3.8 or more to be run. If you don't have it installed on your machine, follow \
the instructions `here <https://github.com/MahdiAll99/MEDimage/blob/main/python.md>`__.

Install via pip
---------------
``MEDimage`` is available on PyPi for installation via ``pip`` which allows you to install the package in one step ::

    pip install medimage-pkg

Install from source
-------------------

1. **Using Conda** |conda-logo|

In order to install the package using conda, make sure to have Anaconda distribution on your machine, you can download and install it by \
following the instructions `here <https://docs.anaconda.com/anaconda/install/index.html>`__.

.. note::
    We recommend updating conda before installing the environnement by running :: 
        
        conda update --yes --name base --channel defaults conda

* Cloning the repository ::

    git clone https://github.com/MahdiAll99/MEDimage.git

* Access the package folder ::

    cd MEDimage

* Using anaconda distribution, we will create and activate the medimage environment. You can do so by simply running ::

    conda env create --name medimage --file environment.yml

* Active the installed environment ::

    conda activate medimage

* If you want to run the notebooks, you must add your installed environnement to jupyter notebook kernels :: 
     
    python -m ipykernel install --user --name=medimage

.. |conda-logo| image:: https://www.psych.mcgill.ca/labs/mogillab/anaconda2/pkgs/anaconda-navigator-1.4.3-py27_0/lib/python2.7/site-packages/anaconda_navigator/static/images/anaconda-icon-1024x1024.png
    :width: 3%
    :target: https://docs.anaconda.com/anaconda/install/index.html

2. **Using Poetry** |poetry-logo|

* Download and install poetry ::

    pip install poetry

More downloading methods can be found `here <https://python-poetry.org/docs/#installation>`__.

* Cloning the repository ::

    git clone https://github.com/MahdiAll99/MEDimage.git

* Access the package folder ::

    cd MEDimage

* Poetry will automatically create a new environment and download the required dependencies after running ::

    poetry install

The created environment will be activated automatically.

* If you wanna run notebooks later, add your installed environnement to jupyter notebook kernels :: 
     
    poetry run python -m ipykernel install --user --name={potry_env_name}

.. note::
    You can use this following command to get information about the currently activated virtual environment ::
        
        poetry env info

.. |poetry-logo| image:: https://python-poetry.org/images/logo-origami.svg
    :width: 3%
    :target: https://python-poetry.org/docs/

Now that you have successfully installed the package, we invite you to follow these :doc:`../tutorials` to further comprehend how to use it.

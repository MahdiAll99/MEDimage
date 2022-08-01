Installation
============

Python installation
-------------------

The MEDimage package requires python 3.8 or more to be run. If you don't have it installed on your machine, follow \
the instructions `here <https://github.com/MahdiAll99/MEDimage/blob/main/python.md>`__.

Cloning the repository
----------------------

In your terminal, clone the repository::

    git clone https://github.com/MahdiAll99/MEDimage.git

Then access the package directory using::

    `cd MEDimage` 

Making the environment
-----------------------

In order to use the package, make sure to have Anaconda distribution on your machine, you can download and install it by \
following the instructions on this `link <https://docs.anaconda.com/anaconda/install/index.html>`__.

Using anaconda distribution, we will create and activate the medimage environment. You can do so by running this command::

    `make -f Makefile.mk create_environment`

This command will install all the dependencies required. And now we initialize conda with::

    `conda init``

And activate the medimage environment using::

    `conda activate medimage`

Once the environment is activated, you can generate the documentation in the doc section or start running the IBSI-Tests \
without documentation (not recommended).
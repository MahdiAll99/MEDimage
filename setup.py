import sys

from setuptools import find_packages, setup

# Check if current python installation is >= 3.8
if sys.version_info < (3, 8, 0):
  raise Exception("MEDimage requires python 3.8 or later")

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name="MEDimage",
    version="0.9.4",
    author="MEDomics consortium",
    author_email="medomics.info@gmail.com",
    description="Python Open-source package for medical images processing and radiomic features extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MahdiAll99/MEDimage",
    project_urls={
        'Documentation': 'https://medimage.readthedocs.io/en/latest/index.html',
        'Github': 'https://github.com/MahdiAll99/MEDimage'
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='radiomics cancer imaging medical research computational imaging',
    python_requires='>=3.8,<=3.10',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=requirements
)

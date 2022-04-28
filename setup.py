import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medimage",
    version="0.0.1",
    author="MEDomicsLab consortium",
    author_email="",
    description="MEDimage package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MahdiAll99/MEDimage",
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    license='GNU General Public License version 3 or any later version',
)

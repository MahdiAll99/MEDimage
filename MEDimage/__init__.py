import logging

from . import utils
from . import processing
from . import biomarkers
from . import filters
from . import wrangling
from . import learning
from .MEDscan import MEDscan


stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
logging.getLogger(__name__).addHandler(stream_handler)

__author__ = "MEDomicsLab consortium"
__version__ = "0.9.4"
__copyright__ = "Copyright (C) MEDomicsLab consortium"
__license__ = "GNU General Public License 3.0"
__maintainer__ = "MAHDI AIT LHAJ LOUTFI"
__email__ = "medomics.info@gmail.com"

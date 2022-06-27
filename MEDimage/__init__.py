import logging

<<<<<<< HEAD
from . import biomarkers 
from . import processing
from . import utils
=======
import MEDimage.biomarkers
import MEDimage.filter
import MEDimage.processing
import MEDimage.utils
>>>>>>> 375738491562ce44a3931c582c27a1071ecacb8e


stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
logging.getLogger(__name__).addHandler(stream_handler)

__author__ = "MEDomicsLab consortium"
__version__ = "0.2.0"
__copyright__ = "Copyright (C) MEDomicsLab consortium"
__license__ = "GNU General Public License 3.0"
__maintainer__ = "MAHDI AIT LHAJ LOUTFI"
__email__ = "Mahdi.Ait.Lhaj.Loutfi@USherbrooke.ca"

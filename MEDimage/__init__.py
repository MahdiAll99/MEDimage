import logging

import MEDimage.biomarkers
import MEDimage.filter
import MEDimage.processing
import MEDimage.utils


stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
logging.getLogger(__name__).addHandler(stream_handler)

__author__ = "MEDomicsLab consortium"
__version__ = "0.2.0"
__copyright__ = "Copyright (C) MEDomicsLab consortium"
__license__ = "GNU General Public License 3.0"
__maintainer__ = "MAHDI AIT LHAJ LOUTFI"
__email__ = "Mahdi.Ait.Lhaj.Loutfi@USherbrooke.ca"

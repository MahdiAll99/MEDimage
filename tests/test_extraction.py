import os
import sys

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath('./MEDimage/'))
sys.path.append(MODULE_DIR)

import MEDimage


class TestExtraction:
        
    def _get_phantom(self):
        phantom = np.zeros((64,64,64))
        phantom[32,32,32] = 255

        return phantom
    
    def _get_random_roi(self):
        roi = np.zeros((64,64,64))
        roi[
            np.random.randint(0,64,5),
            np.random.randint(0,64,5),
            np.random.randint(0,64,5)] = 1
        return roi
    
    def test_morph_features(self):
        phantom = self._get_phantom()
        roi = self._get_random_roi()
        morph = MEDimage.biomarkers.morph.extract_all(
            vol=phantom, 
            mask_int=roi, 
            mask_morph=roi,
            res=[1,1,1],
            intensity_type="arbitrary"
        )
        morph_vol = MEDimage.biomarkers.morph.vol(
            vol=phantom, 
            mask_int=roi, 
            mask_morph=roi,
            res=[1,1,1]
        )
        assert morph_vol == morph["Fmorph_vol"]
        assert round(morph_vol, 2) == 0.83

    def test_stats_features(self):
        phantom = self._get_phantom()
        roi = self._get_random_roi()
        vol_int_re = MEDimage.processing.roi_extract(
            vol=phantom, 
            roi=roi
        )
        stats = MEDimage.biomarkers.stats.extract_all(
            vol=vol_int_re,
            intensity_type="arbitrary"
        )
        kurt = MEDimage.biomarkers.stats.kurt(
            vol=vol_int_re,
        )
        assert kurt == stats["Fstat_kurt"]
        assert round(kurt, 2) == -3.0

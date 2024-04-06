import os
import sys

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath('./MEDimage/'))
sys.path.append(MODULE_DIR)

import MEDimage


class TestExtraction:
    """
    Test the extraction of morphological and statistical features.

    Features are extracted from the IBSI phantom, and values are compared to the IBSI ones.

    Phantom and reference values can be found in the IBSI manual: https://arxiv.org/pdf/1612.07003.pdf
    """
    def __get_phantom(self):
        phantom = np.array(
            [[
                [1, 1, 1, 1,],
                [1, 1, 1, 1,],
                [4, 1, 1, 1,],
                [4, 4, 1, 1,]],

                [[4, 4, 4, 4,],
                [4, 1, 1, 1,],
                [1, 1, 1, 1,],
                [4, 4, 1, 1,]],

                [[4, 4, 4, 4,],
                [6, 6, 1, 1,],
                [6, 3, 9, 1,],
                [6, 6, 6, 6,]],

                [[1, 1, 1, 1,],
                [1, 1, 1, 1,],
                [4, 1, 1, 1,],
                [4, 1, 1, 1,]],

                [[1, 1, 1, 1,],
                [1, 1, 1, 1,],
                [1, 1, 1, 1,],
                [1, 1, 1, 1,]]], dtype=np.float32
        )

        return phantom
    
    def __get_random_roi(self):
        roi = np.array(
            [[
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1]],

                [[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]],

                [[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1]],

                [[1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]],

                [[1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]]], dtype=np.int16
        )

        return roi
    
    def test_morph_features(self):
        phantom = self.__get_phantom()
        roi = self.__get_random_roi()
        morph = MEDimage.biomarkers.morph.extract_all(
            vol=phantom, 
            mask_int=roi, 
            mask_morph=roi,
            res=[2, 2, 2],
            intensity_type="arbitrary"
        )
        morph_vol = MEDimage.biomarkers.morph.vol(
            vol=phantom, 
            mask_int=roi, 
            mask_morph=roi,
            res=[2, 2, 2]
        )
        surface_area = MEDimage.biomarkers.morph.area(
            vol=phantom, 
            mask_int=roi, 
            mask_morph=roi,
            res=[2, 2, 2]
        )
        assert morph_vol == morph["Fmorph_vol"]
        assert abs(morph_vol - 556) < 1
        assert surface_area == morph["Fmorph_area"]
        assert abs(surface_area - 388) < 1

    def test_stats_features(self):
        phantom = self.__get_phantom()
        roi = self.__get_random_roi()
        vol_int_re = MEDimage.processing.roi_extract(
            vol=phantom, 
            roi=roi
        )
        stats = MEDimage.biomarkers.stats.extract_all(
            vol=vol_int_re,
            intensity_type="definite"
        )
        kurt = MEDimage.biomarkers.stats.kurt(
            vol=vol_int_re,
        )
        skewness = MEDimage.biomarkers.stats.skewness(
            vol=vol_int_re,
        )
        assert kurt == stats["Fstat_kurt"]
        assert abs(kurt + 0.355) < 0.01
        assert skewness == stats["Fstat_skew"]
        assert abs(skewness - 1.08) < 0.01

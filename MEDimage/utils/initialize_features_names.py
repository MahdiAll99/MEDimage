#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict, List, Tuple


def initialize_features_names(image_space_struct: Dict) -> Tuple[List, List]:
    """Finds all the features names from `image_space_struct`

    Args:
        image_space_struct(Dict): Dictionary of the extracted features (Texture & Non-texture)
    
    Returns:
        Tuple[List, List]: Two lists of the texture and non-texture features names found in the `image_space_struct`.
    """
    # First entry is the names of feature types. Second entry is the name of
    # the features for a given feature type. Third entry is the name of the
    # extraction parameters for all features of a given feature type.
    non_text_cell = [0] * 3
    # First entry is the names of feature types. Second entry is the name of
    # the features for a given feature type. Third entry is the name of the
    # extraction parameters for all features of a given feature type.
    text_cell = [0] * 3

    # NON-TEXTURE FEATURES
    field_non_text = [key for key in image_space_struct.keys() if key != 'texture']
    n_non_text_type = len(field_non_text)
    non_text_cell[0] = field_non_text
    non_text_cell[1] = [0] * n_non_text_type
    non_text_cell[2] = [0] * n_non_text_type

    for t in range(0, n_non_text_type):
        dic_image_space_struct_non_text = image_space_struct[non_text_cell[0][t]]
        field_params_non_text = [
            key for key in dic_image_space_struct_non_text.keys()]
        dic_image_space_struct_params_non_text = image_space_struct[non_text_cell[0]
                                                               [t]][field_params_non_text[0]]
        field_feat_non_text = [
            key for key in dic_image_space_struct_params_non_text.keys()]
        non_text_cell[1][t] = field_feat_non_text
        non_text_cell[2][t] = field_params_non_text

    # TEXTURE FEATURES
    dic_image_space_struct_texture = image_space_struct['texture']
    field_text = [key for key in dic_image_space_struct_texture.keys()]
    n_text_type = len(field_text)
    text_cell[0] = field_text
    text_cell[1] = [0] * n_text_type
    text_cell[2] = [0] * n_text_type

    for t in range(0, n_text_type):
        dic_image_space_struct_text = image_space_struct['texture'][text_cell[0][t]]
        field_params_text = [key for key in dic_image_space_struct_text.keys()]
        dic_image_space_struct_params_text = image_space_struct['texture'][text_cell[0]
                                                                       [t]][field_params_text[0]]
        field_feat_text = [
            key for key in dic_image_space_struct_params_text.keys()]
        text_cell[1][t] = field_feat_text
        text_cell[2][t] = field_params_text

    return non_text_cell, text_cell

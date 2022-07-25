#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import pathlib
from typing import Dict


def _is_jsonable(data: any) -> bool:
    """Checks if the given ``data`` is JSON serializable.

    Args:
        data (Any): ``Data`` that will be checked.

    Returns:
        bool: True if the given ``data`` is serializable, False if not.
    """
    try:
        json.dumps(data)
        return True
    except (TypeError, OverflowError):
        return False


def posix_to_string(dictionnary: Dict) -> None:
    """Converts all Pathlib.Path to str [Pathlib is not serializable].

    Args:
        dictionnary (Dict): ``Input`` dict with Pathlib.Path values to convert.

    Returns:
        None: Mutate the given dictionary.
    """
    for key, value in dictionnary.items():
        if type(value) is dict:
            posix_to_string(value)
        else:
            dictionnary[key] = str(value) if isinstance(value, pathlib.PosixPath) else value


def load_json(file_path: pathlib.Path) -> Dict:
    """Wrapper to json.load function.

    Args:
        file_path (Path): Path of the json file to load.

    Returns:
        Dict: The loaded json file.
     
    """
    with open(file_path, 'r') as fp:
        return json.load(fp)


def save_json(file_path: pathlib.Path, data: any, cls=None) -> None:
    """Wrapper to json.dump function.

    Args:
        file_path (Path): Path to write the json file to.
        data (Any): Data to write to the given path. Must be serializable by JSON.
        cls(object, optional): Costum JSONDecoder subclass. If not specified JSONDecoder is used. 

    Returns:
        None: saves the ``data`` in JSON file to the ``file_path``.

    Raises:
        TypeError: If ``data`` is not JSON serializable.
    """
    if _is_jsonable(data):
        with open(file_path, 'w') as fp:
            json.dump(data, fp, indent=4, cls=cls)
    else:
        raise TypeError("The given data is not JSON serializable. \
            We rocommend using a costum encoder.")

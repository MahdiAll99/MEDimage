import json
import pathlib


def posix_to_string(dictionnary):
    """
    function that converts all Pathlib.Path to str [json can't serialize Pathlib ]
    :param dictionnary: the dictionary containing to converts
    :return: None, mutate the given dictionary
    """
    for key, value in dictionnary.items():
        if type(value) is dict:
            posix_to_string(value)
        else:
            dictionnary[key] = str(value) if isinstance(value, pathlib.PosixPath) else value


def loadjson(filepath):
    """
    wrapper to json.load function
    :param filepath: the path of the json file to load
    :return: the loaded json file
    """
    with open(filepath, 'r') as fp:
        return json.load(fp)


def savejson(filepath, data):
    """
    wrapper to json.dump function
    :param filepath: the path to write the json file to.
    :param data: the data to write to the given path
    :return: None
    """
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


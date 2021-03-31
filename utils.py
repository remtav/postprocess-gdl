import warnings
from typing import List

import torch
from ruamel_yaml import YAML


def read_parameters(param_file):
    """Read and return parameters in .yaml file
    Args:
        param_file: Full file path of the parameters file
    Returns:
        YAML (Ruamel) CommentedMap dict-like object
    """
    yaml = YAML()
    with open(param_file) as yamlfile:
        params = yaml.load(yamlfile)
    return params


def get_key_def(key, config, default=None, msg=None, delete=False, expected_type=None):
    """Returns a value given a dictionary key, or the default value if it cannot be found.
    :param key: key in dictionary (e.g. generated from .yaml)
    :param config: (dict) dictionary containing keys corresponding to parameters used in script
    :param default: default value assigned if no value found with provided key
    :param msg: message returned with AssertionError si length of key is smaller or equal to 1
    :param delete: (bool) if True, deletes parameter, e.g. for one-time use.
    :return:
    """
    if not config:
        return default
    elif isinstance(key, list):  # is key a list?
        if len(key) <= 1:  # is list of length 1 or shorter? else --> default
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("Must provide at least two valid keys to test")
        for k in key:  # iterate through items in list
            if k in config:  # if item is a key in config, set value.
                val = config[k]
                if delete:  # optionally delete parameter after defining a variable with it
                    del config[k]
        val = default
    else:  # if key is not a list
        if key not in config or config[key] is None:  # if key not in config dict
            val = default
        else:
            val = config[key] if config[key] != 'None' else None
            if expected_type and val is not False:
                assert isinstance(val, expected_type), f"{val} is of type {type(val)}, expected {expected_type}"
            if delete:
                del config[key]
    return val


def load_checkpoint(filename):
    ''' Loads checkpoint from provided path
    :param filename: path to checkpoint as .pth.tar or .pth
    :return: (dict) checkpoint ready to be loaded into model instance
    '''
    try:
        print(f"=> loading model '{filename}'\n")
        # For loading external qgis_models with different structure in state dict.
        checkpoint = torch.load(filename, map_location='cpu')
        if 'model' not in checkpoint.keys():
            # Place entire state_dict inside 'model' key
            temp_checkpoint = {'model': {k: v for k, v in checkpoint.items()}}
            del checkpoint
            checkpoint = temp_checkpoint
        return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"=> No model found at '{filename}'")


def compare_config_yamls(yaml1: dict, yaml2: dict, update_yaml1: bool = False) -> List:
    """
    Checks if values for same keys or subkeys (max depth of 2) of two dictionaries match.
    :param yaml1: (dict) first dict to evaluate
    :param yaml2: (dict) second dict to evaluate
    :param update_yaml1: (bool) it True, values in yaml1 will be replaced with values in yaml2,
                         if the latters are different
    :return: dictionary of keys or subkeys for which there is a value mismatch if there is, or else returns None
    """
    if not (isinstance(yaml1, dict) or isinstance(yaml2, dict)):
        raise TypeError(f"Expected both yamls to be dictionaries. \n"
                        f"Yaml1's type is  {type(yaml1)}\n"
                        f"Yaml2's type is  {type(yaml2)}")
    for section, params in yaml2.items():  # loop through main sections of config yaml ('global', 'sample', etc.)
        if section not in yaml1.keys():  # create key if not in dictionary as we loop
            yaml1[section] = {}
        for param, val2 in params.items():  # loop through parameters of each section ('samples_size','debug_mode',...)
            if param not in yaml1[section].keys():  # create key if not in dictionary as we loop
                yaml1[section][param] = {}
            # set to None if no value for that key
            val1 = get_key_def(param, yaml1[section], default=None)
            if isinstance(val2, dict):  # if value is a dict, loop again to fetch end val (only recursive twice)
                for subparam, subval2 in val2.items():
                    if subparam not in yaml1[section][param].keys():  # create key if not in dictionary as we loop
                        yaml1[section][param][subparam] = {}
                    # set to None if no value for that key
                    subval1 = get_key_def(subparam, yaml1[section][param], default=None)
                    if subval2 != subval1:
                        # if value doesn't match between yamls, emit warning
                        warnings.warn(f"YAML value mismatch: section \"{section}\", key \"{param}/{subparam}\"\n"
                                        f"Current yaml value: \"{subval1}\"\nHDF5s yaml value: \"{subval2}\"\n")
                        if update_yaml1:  # update yaml1 with subvalue of yaml2
                            yaml1[section][param][subparam] = subval2
                            warnings.warn(f'Value in yaml1 updated')
            elif val2 != val1:
                warnings.warn(f"YAML value mismatch: section \"{section}\", key \"{param}\"\n"
                                f"Current yaml value: \"{val2}\"\nHDF5s yaml value: \"{val1}\"\n"
                                f"Problems may occur.")
                if update_yaml1:  # update yaml1 with value of yaml2
                    yaml1[section][param] = val2
                    print(f'Value in yaml1 updated')

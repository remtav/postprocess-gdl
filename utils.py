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
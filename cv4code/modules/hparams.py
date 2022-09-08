# SPDX-License-Identifier: Apache-2.0
import yaml

class YHparams:
    """
    A utility class used for yaml controlled hyperparams

    access {key : value} from yaml as class attributes
    """
    def __init__(self, filepath, tag='default'):
        with open(filepath, 'r') as fd:
            conf = yaml.load(fd, Loader=yaml.FullLoader)
        
        if tag not in conf:
            raise ValueError(f'{tag} does not exist in the loaded yaml at {filepath}')

        conf = conf[tag]
        for k, v in conf.items():
            setattr(self, k, v)
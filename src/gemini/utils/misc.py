#!/usr/bin/env python

#==============================================================================

def get_args(self, exclude=['self', '__class__', 'kwargs'], **kwargs):
    args = {key: kwargs[key] for key in kwargs if not key in exclude}
    if 'kwargs' in kwargs:
        for key, val in kwargs['kwargs'].items():
            args[key] = val
    return args

#==============================================================================

"""
Events data types hierarchy:
- Base > Linear > Spatial
- Base > Point > Spatial

Terminology
- Match: Where keys match between two events
- Intersect: Where bounds intersect between two matched events
"""

from functools import wraps


def deprecate_geometry(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        obj = args[0]
        return fn(*args, **kwargs)
    return wrapper

def modify_geometry(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        obj = args[0]
        return fn(*args, **kwargs)
    return wrapper


class EventsCollection_Base(object):
    """
    EventsCollection subclass with basic features applicable to all events 
    data types.
    """

    def get_group(self):
        pass

    def get_subset(self):
        pass

    def get_matching(self):
        pass
    
    def merge(self): # behavior depends on nature of other collection
        pass

    def intersect(self): # intersecting
        pass

    def overlay(self):
        pass

    @deprecate_geometry
    def round(self):
        pass

    @deprecate_geometry
    def clip(self):
        pass

    @deprecate_geometry
    def shift(self):
        pass

    @deprecate_geometry
    def separate(self):
        pass


class EventsCollection_Linear(object):
    """
    EventsCollection subclass with additional features for linear events data 
    types.
    """
    
    @modify_geometry
    def dissolve(self):
        pass

    @deprecate_geometry
    def grid(self): # to_grid, spatial component
        pass
    
    @deprecate_geometry
    def resegment(self): # to_windows, spatial component
        pass


class EventsCollection_SpatialLinear(object):
    """
    EventsCollection subclass with features for events data with a spatial 
    component that is compatible with the nature of the tabular component.
    That is, linear events collections must contain linear geometries (i.e., 
    LineStrings or MultiLineStrings), while point events collections must 
    contain point geometries (i.e., Points).
    """
    
    def project(self):
        pass

    def project_parallel(self):
        pass
    
class EventsCollection_SpatialPoint(object):
    """
    EventsCollection subclass with features for events data with a spatial 
    component that is compatible with the nature of the tabular component.
    That is, linear events collections must contain linear geometries (i.e., 
    LineStrings or MultiLineStrings), while point events collections must 
    contain point geometries (i.e., Points).
    """
    
    def project(self):
        pass





class EventsCollection(object):

    _default_keys = ['key']
    _default_beg = ['beg']
    _default_end = ['end']
    _default_loc = ['loc']

    def __init__(self):
        pass

    @classmethod
    def register_defaults(cls, keys=None, beg=None, end=None, loc=None):
        """
        Register new defaults target dataframe labels. These will take 
        precedence over standard defaults and previously registered defaults.
        """
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            cls._default_keys = keys + cls._default_keys
        if beg is not None:
            if isinstance(beg, str):
                beg = [beg]
            cls._default_beg = beg + cls._default_beg
        if end is not None:
            if isinstance(end, str):
                end = [end]
            cls._default_end = end + cls._default_end
        if loc is not None:
            if isinstance(loc, str):
                loc = [loc]
            cls._default_loc = loc + cls._default_loc
            
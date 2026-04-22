# Expose core classes
from linref.events.base import EventsData
from linref.ext.base import LRS, LRS_Accessor

# Expose additional features
from linref.ext.base import integrate
from linref.utility.direction import extract_direction, extract_bearing

# Expose datasets module
from linref import datasets

# Expose extension settings
from linref.options import options, set_default_lrs

__all__ = [
    'EventsData',
    'LRS',
    'LRS_Accessor',
    'integrate',
    'extract_direction',
    'extract_bearing',
    'datasets',
    'options',
    'set_default_lrs',
]
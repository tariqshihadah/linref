# Expose core classes
from linref.events.base import EventsData
from linref.ext.base import LRS, LRS_Accessor

# Expose additional features
from linref.ext.base import integrate
from linref.ext.spatial import generate_intersection_pairs, generate_intersection_nodes
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
    'generate_intersection_pairs',
    'generate_intersection_nodes',
    'extract_direction',
    'extract_bearing',
    'datasets',
    'options',
    'set_default_lrs',
]
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("linref")
except PackageNotFoundError:
    __version__ = "unknown"

# Expose core classes
from linref.events.base import EventsData
from linref.ext.lrs import LRS
from linref.ext.base import LRS_Accessor

# Expose additional features
from linref.ext.base import integrate
from linref.ext.spatial import generate_intersection_pairs, generate_intersection_nodes
from linref.utility.direction import extract_direction, extract_bearing

# Expose datasets module
from linref import datasets

# Expose extension settings
from linref.options import options, set_default_lrs

# Expose errors
from linref.errors import RemovedFeatureError


# Removed class stub (v0.1.x -> v1.0 migration)
class EventsCollection:
    """Removed in v1.0. Use the DataFrame.lr accessor instead."""

    def __init__(self, *args, **kwargs):
        raise RemovedFeatureError(
            "linref.EventsCollection has been removed in linref v1.0.0.\n\n"
            "The v1.0.0 release is a complete redesign using the pandas accessor\n"
            "pattern (df.lr). See the migration guide:\n"
            "  https://linref.readthedocs.io/en/latest/migration.html\n\n"
            "To continue using the old API, pin to: linref<1.0"
        )

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
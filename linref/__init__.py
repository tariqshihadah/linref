# Expose core classes
from linref.events.base import EventsData
from linref.ext.base import LRS, LRS_Accessor

# Expose additional features
from linref.ext.base import integrate
from linref.utility.direction import extract_direction, extract_bearing

# Expose datasets module
from linref import datasets

# Expose extension settings
set_default_lrs = LRS_Accessor.set_default_lrs
set_default_geometry_sync = LRS_Accessor.set_default_geometry_sync
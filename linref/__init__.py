# Expose core classes
from linref.events.base import EventsData
from linref.ext.base import LRS, LRS_Accessor

# Expose additional features
from linref.ext.base import integrate

# Expose extension settings
set_default_lrs = LRS_Accessor.set_default_lrs
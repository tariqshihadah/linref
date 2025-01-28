from linref.ext.base import LRS


#
# Define default LRS instances
#

LRS(
    keys_col='key',
    locs_col='loc',
    begs_col='beg',
    ends_col='end',
    geom_col='geom',
    closed='left_mod',
)
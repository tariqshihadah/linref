"""
Shared setup for linref example notebooks.

Import this module at the top of any example notebook to get pre-loaded
datasets with the default LRS already configured::

    from _setup import lr, roadways, crashes, pavement

This avoids duplicating boilerplate across notebooks while keeping each
notebook focused on its topic.
"""
import linref as lr

# Configure default LRS once
lr.set_default_lrs(
    key_col=['route'],
    loc_col='loc',
    beg_col='beg',
    end_col='end',
    geom_col='geometry',
    geom_m_col='geometry_m',
    closed='left_mod',
)

# Load sample datasets (LRS is applied via the default set above)
roadways = lr.datasets.load('roadways')
crashes = lr.datasets.load('crashes')
pavement = lr.datasets.load('pavement')

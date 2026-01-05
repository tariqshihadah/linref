"""
Dataset loaders for linref toy datasets.

Provides easy access to sample datasets for learning and testing
linear referencing operations.
"""

import pandas as pd
from pathlib import Path
from typing import Union
import geopandas as gpd


def _get_data_path(filename: str) -> Path:
    """Get the absolute path to a data file."""
    return Path(__file__).parent / '_data' / filename


def load(name: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Load a toy dataset with LRS pre-configured.
    
    Available datasets:
    - 'roadways': Roadway linear events with geometry (10 segments)
    - 'crashes': Crash point events with geometry (20 crashes)
    - 'pavement': Pavement condition linear events (14 segments, no geometry)
    
    Parameters
    ----------
    name : str
        Name of the dataset to load. Options: 'roadways', 'crashes', 'pavement'.
    
    Returns
    -------
    DataFrame or GeoDataFrame
        Dataset with LRS already configured. Returns GeoDataFrame for datasets
        with geometry (roadways, crashes), DataFrame for others (pavement).
    
    Examples
    --------
    >>> import linref as lr
    >>> 
    >>> # Load datasets
    >>> roads = lr.datasets.load('roadways')
    >>> crashes = lr.datasets.load('crashes')
    >>> pavement = lr.datasets.load('pavement')
    >>> 
    >>> # Use immediately - LRS already configured
    >>> dissolved = roads.lr.dissolve(retain=['speed_limit'])
    >>> relation = roads.lr.relate(crashes)
    >>> roads['crash_count'] = relation.count()
    """
    # Import here to avoid circular imports
    from linref import LRS
    
    if name == 'roadways':
        # Load roadways GeoJSON
        data_path = _get_data_path('roadways.geojson')
        gdf = gpd.read_file(data_path)
        gdf = gdf.lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            geom_col='geometry',
            closed='right'
        )
        return gdf
        
    elif name == 'crashes':
        # Load crashes GeoJSON
        data_path = _get_data_path('crashes.geojson')
        gdf = gpd.read_file(data_path)
        gdf = gdf.lr.set_lrs(
            key_col=['route'],
            loc_col='location',
            geom_col='geometry'
        )
        return gdf
        
    elif name == 'pavement':
        # Load pavement CSV
        data_path = _get_data_path('pavement.csv')
        df = pd.read_csv(data_path)
        df = df.lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='right'
        )
        return df
        
    else:
        available = ['roadways', 'crashes', 'pavement']
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {', '.join(available)}"
        )


def list_datasets() -> pd.DataFrame:
    """
    List all available toy datasets with descriptions.
    
    Returns
    -------
    DataFrame
        Table with dataset names, types, and descriptions.
    
    Examples
    --------
    >>> import linref as lr
    >>> print(lr.datasets.list_datasets())
    """
    datasets = {
        'name': ['roadways', 'crashes', 'pavement'],
        'type': ['Linear Events', 'Point Events', 'Linear Events'],
        'records': [10, 20, 14],
        'has_geometry': [True, True, False],
        'description': [
            'Roadway segments with traffic and speed attributes',
            'Crash point locations with severity and mode',
            'Pavement condition segments with ratings and types'
        ]
    }
    return pd.DataFrame(datasets)

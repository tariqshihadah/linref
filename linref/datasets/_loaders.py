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


def load(name: str, set_lrs: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Load a toy dataset.
    
    Available datasets:
    - 'roadways': Roadway linear events with geometry (10 segments)
    - 'crashes': Crash point events with geometry (20 crashes)
    - 'pavement': Pavement condition linear events (14 segments, no geometry)
    
    Parameters
    ----------
    name : str
        Name of the dataset to load. Options: 'roadways', 'crashes', 'pavement'.
    set_lrs : bool, default False
        Whether to automatically configure LRS on the loaded dataset. When True,
        configures LRS with standard column names (route, loc, beg, end, geometry,
        geometry_m, closed='left_mod').
    
    Returns
    -------
    DataFrame or GeoDataFrame
        Dataset with or without LRS configured based on set_lrs parameter. 
        Returns GeoDataFrame for datasets with geometry (roadways, crashes), 
        DataFrame for others (pavement).
    
    Examples
    --------
    >>> import linref as lr
    >>> 
    >>> # Load datasets without LRS (default)
    >>> roads = lr.datasets.load('roadways')
    >>> 
    >>> # Load with LRS pre-configured
    >>> roads = lr.datasets.load('roadways', set_lrs=True)
    >>> dissolved = roads.lr.dissolve(retain=['speed_limit'])
    >>> 
    >>> # Or configure LRS manually
    >>> roads = lr.datasets.load('roadways')
    >>> roads = roads.lr.set_lrs(
    ...     key_col=['route'], beg_col='beg', end_col='end',
    ...     geom_col='geometry', closed='left_mod'
    ... )
    """
    if name == 'roadways':
        # Load roadways GeoJSON
        data_path = _get_data_path('roadways.geojson')
        gdf = gpd.read_file(data_path)
        
        if set_lrs:
            gdf = gdf.lr.set_lrs(
                key_col=['route'],
                loc_col='loc',
                beg_col='beg',
                end_col='end',
                geom_col='geometry',
                geom_m_col='geometry_m',
                closed='left_mod'
            )
        return gdf
        
    elif name == 'crashes':
        # Load crashes GeoJSON
        data_path = _get_data_path('crashes.geojson')
        gdf = gpd.read_file(data_path)
        
        if set_lrs:
            gdf = gdf.lr.set_lrs(
                key_col=['route'],
                loc_col='loc',
                geom_col='geometry',
                geom_m_col='geometry_m'
            )
        return gdf
        
    elif name == 'pavement':
        # Load pavement CSV
        data_path = _get_data_path('pavement.csv')
        df = pd.read_csv(data_path)
        
        if set_lrs:
            df = df.lr.set_lrs(
                key_col=['route'],
                loc_col='loc',
                beg_col='beg',
                end_col='end',
                geom_m_col='geometry_m',
                closed='left_mod'
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

import geopandas as gpd
from shapely.ops import nearest_points


def join_nearby(left, right, buffer=100, lsuffix='left', rsuffix='right', 
                choose='min', dist_label='DISTANCE'):
    """
    Join a geodataframe to the nearest, furthest, or all records from a second 
    geodataframe within a defined buffer area.

    Modified: 2020-09-21
    """
    # Collect geodataframe information
    index_left_orig  = left.index.name
    index_right_orig = right.index.name
    index_left  = 'index_left'  if not index_left_orig  else index_left_orig
    index_right = 'index_right' if not index_right_orig else index_right_orig

    # Perform spatial join with the given buffer
    joined = gpd.sjoin(left.set_geometry(left.geometry.buffer(buffer)), right,
                       how='left', lsuffix=lsuffix, rsuffix=rsuffix)

    # Restore original geometry and index names
    joined.geometry = left.geometry
    joined = joined.rename(
        columns={'index_left': index_left, 'index_right': index_right})

    # Define distance computation function
    def _get_distance(o1, o2):
        # If geometries are valid, compute distance
        try:
            p1, p2 = nearest_points(o1, o2)
            return p1.distance(p2)
        except (AttributeError, ValueError) as e:
            return -1
            
    # Compute distances
    geoms = joined[[left.geometry.name,index_right]].merge(
        right.geometry, left_on=index_right, right_index=True, how='left')
    points = zip(geoms.iloc[:,0], geoms.iloc[:,2])
    joined[dist_label] = list(map(lambda x: _get_distance(*x), points))
    
    # Choose resulting records
    joined = joined.rename_axis(index=index_left) \
        .sort_values(by=[index_left,dist_label], ascending=[True,True])
    joined = joined.rename_axis(index=index_left_orig)
    if choose == 'min':
        joined = joined[~joined.index.duplicated(keep='first')]
    elif choose == 'max':
        joined = joined[~joined.index.duplicated(keep='last')]
    elif choose == 'all':
        pass
    else:
        raise ValueError("Choose parameter must be 'min', 'max', or 'all'.")
    return joined


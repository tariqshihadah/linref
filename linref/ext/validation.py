from __future__ import annotations
import functools
import warnings
from linref.errors import GeometrySyncWarning, GeometrySyncError, LinrefError


def _validate_lrs(*components: str):
    """Validate LRS components before executing an accessor method.

    Each component name is resolved to a corresponding ``validate_<name>()``
    method on the accessor. Validation uses each method's default options and
    raises the same Linref exception type with additional context naming the
    decorated method and its requirements when validation fails.

    Parameters
    ----------
    *components : str
        Names of accessor validation methods without the ``validate_`` prefix,
        such as ``'linear'`` or ``'spatial_m'``.

    Returns
    -------
    callable
        A decorator that validates the requested components before calling the
        wrapped method.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            labels = {
                'grouped': 'grouped data',
                'located': 'location measures',
                'point': 'point-event data',
                'linear': 'linear-event data',
                'spatial': 'Shapely geometries',
                'spatial_m': 'M-enabled geometries',
                'chained': 'chain data',
            }
            requirements = [labels.get(name, name) for name in components]
            if len(requirements) == 1:
                required = requirements[0]
            elif len(requirements) == 2:
                required = ' and '.join(requirements)
            else:
                required = ', '.join(requirements[:-1]) + ', and ' + requirements[-1]
            for component in components:
                try:
                    getattr(self, f'validate_{component}')()
                except LinrefError as error:
                    raise type(error)(
                        f"The `{func.__name__}` method requires valid "
                        f"{required}. Validation failed: {error}"
                    ) from error
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def _method_deprecates_geometry(func) -> callable:
    """
    Decorator for LRS_Accessor methods that may de-synchronize the geometry
    of the underlying GeoDataFrame. Deals with this by raising an error,
    raising a warning, or dropping the geometry column, depending on the
    user's preference.

    Parameters
    ----------
    func : callable
        The LRS_Accessor method to decorate.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if the dataframe is spatial
        if self.is_spatial or self.is_spatial_m:
            # Check for geometry synchronization preference
            geometry_sync = kwargs.pop('geometry_sync', self.geometry_sync)
            # Identify action based on preference
            if geometry_sync == 'error':
                raise GeometrySyncError(
                    f"The `{func.__name__}` method may de-synchronize the "
                    "geometry of the underlying GeoDataFrame with event data. "
                    "To proceed anyway, set the `LRS.geometry_sync` attribute "
                    "to 'warn' or 'drop'."
                )
            elif geometry_sync == 'warn':
                warnings.warn(
                    f"The `{func.__name__}` method may de-synchronize the "
                    "geometry of the underlying GeoDataFrame with event data.",
                    GeometrySyncWarning
                )
            elif geometry_sync in ('none', 'ignore'):
                pass
            elif geometry_sync == 'drop':
                # Drop geometry column
                if self.is_spatial:
                    self.df = self.df.drop(columns=self.lrs.geom_col)
                if self.is_spatial_m:
                    self.df = self.df.drop(columns=self.lrs.geom_m_col)
        return func(self, *args, **kwargs)
    return wrapper
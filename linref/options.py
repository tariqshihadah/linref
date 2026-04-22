from __future__ import annotations
from linref.ext.lrs import LRS


_VALID_GEOMETRY_SYNC = ['none', 'warn', 'error', 'remove']


class _Options:
    """
    Package-level options for linref.

    Attributes
    ----------
    default_lrs : LRS
        The default LRS applied to new DataFrames when no LRS has been set.
    default_geometry_sync : str
        The default geometry synchronization behavior. Must be one of
        'none', 'warn', 'error', or 'remove'.
    """

    def __init__(self):
        self.reset()

    @property
    def default_lrs(self) -> LRS:
        return self._default_lrs

    @default_lrs.setter
    def default_lrs(self, value):
        if not isinstance(value, LRS):
            raise ValueError("default_lrs must be an LRS instance.")
        self._default_lrs = value

    @property
    def default_geometry_sync(self) -> str:
        return self._default_geometry_sync

    @default_geometry_sync.setter
    def default_geometry_sync(self, value: str):
        if value not in _VALID_GEOMETRY_SYNC:
            raise ValueError(
                f"Invalid geometry synchronization behavior '{value}'. "
                f"Must be one of {_VALID_GEOMETRY_SYNC}."
            )
        self._default_geometry_sync = value

    def reset(self):
        """Reset all options to their default values."""
        self._default_lrs = LRS()
        self._default_geometry_sync = 'warn'

    def __repr__(self) -> str:
        return (
            f"linref.options\n"
            f"  default_lrs:           {self._default_lrs!r}\n"
            f"  default_geometry_sync:  {self._default_geometry_sync!r}"
        )


options = _Options()


def set_default_lrs(lrs: LRS | None = None, **kwargs) -> LRS:
    """
    Set the default LRS for the linref package.

    Parameters
    ----------
    lrs : LRS, default None
        The LRS object to set as the default. If None, an LRS will be
        constructed from the provided keyword arguments.
    **kwargs
        Keyword arguments passed to the LRS constructor when ``lrs`` is None.

    Returns
    -------
    LRS
        The LRS object that was set as the default.
    """
    if lrs is None:
        lrs = LRS(**kwargs)
    options.default_lrs = lrs
    return lrs

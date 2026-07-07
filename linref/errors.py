class LinrefError(Exception):
    """Base class for exceptions in Linref."""

class LRSConfigurationError(LinrefError):
    """Exception raised for errors in the LRS configuration."""

class LRSCompatibilityError(LinrefError):
    """Exception raised for LRS compatibility issues between DataFrames."""

class GeometryTopologyError(LinrefError):
    """Exception raised for errors in geometry topology."""

class EventTopologyError(LinrefError):
    """Exception raised for errors in event topology."""

class GeometrySyncError(LinrefError):
    """Exception raised for errors in geometry synchronization."""

class GeometrySyncWarning(Warning):
    """Warning raised for potential issues in geometry synchronization."""

class GeometryMeasureError(LinrefError):
    """Exception raised for errors related to M-enabled geometries."""

class LinrefDeprecationWarning(FutureWarning):
    """Warning for deprecated linref features (shown by default)."""

class RemovedFeatureError(LinrefError):
    """Error raised when accessing features removed in v1.0."""
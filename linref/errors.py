class LinrefError(Exception):
    """Base class for exceptions in Linref."""

class LRSConfigurationError(LinrefError):
    """Exception raised for errors in the LRS configuration."""
import numpy as np

def validate_label(label):
    """
    Validate input as a valid `pandas` column label.

    Parameters
    ----------
    label : label
        The label to validate.

    Returns
    -------
    label
        The validated label.
    """
    if not isinstance(label, (str, int)):
        raise TypeError(
            f"Invalid input label: {label}. Must be a valid pandas column "
            "label.")
    return label

def validate_labels(labels):
    """
    Validate input as a list of valid `pandas` column labels.

    Parameters
    ----------
    labels : array-like
        The labels to validate.

    Returns
    -------
    list
        List of validated labels.
    """
    invalid = []
    for label in labels:
        try:
            validate_label(label)
        except:
            invalid.append(label)
    if len(invalid) > 0:
        raise ValueError(
            f"Invalid input labels: {invalid}. Must be valid pandas column "
            "labels.")
    return labels

def label_or_list(obj):
    """
    Return a list of labels from a single label or list of labels.

    Parameters
    ----------
    obj : label or array-like
        The label or array-like of labels to convert.

    Returns
    -------
    list
        List of labels.
    """
    if not isinstance(obj, (list, np.ndarray)):
        return [validate_label(obj)]
    else:
        return validate_labels(obj)
    
def label_list_or_none(obj, if_none=None):
    """
    Return a list of labels from a single label, list of labels, or None.

    Parameters
    ----------
    obj : label or array-like
        The label or array-like of labels to convert.

    Returns
    -------
    list or None
        List of labels or None.
    """
    if obj is None:
        return if_none
    else:
        return label_or_list(obj)
    
def label_or_none(obj, if_none=None):
    """
    Return a label from a single label or None.

    Parameters
    ----------
    obj : label
        The label to convert.

    Returns
    -------
    label or None
        Label or None.
    """
    if obj is None:
        return if_none
    else:
        return validate_label(obj)
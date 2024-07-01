from typing import Optional, Union
import warnings

import os

import pandas as pd

def normalize_ext(ext: Optional[str], include_period: bool = True) -> Optional[str]:
    """
    Normalize the format of a file extension.

    Parameters
    ----------
    ext : str
        The file extension.
    include_period : bool, default True
        Whether to include a period in the returned extension.

    Returns
    -------
    str or None
        Processed file extension with or without a period, or None if ext is None.
    """
    if ext is None:
        return None

    if ext.startswith("."):
        if include_period:
            # Has period and we want the period
            return ext
        # Has period and we don't want the period
        return ext[1:]

    if include_period:
        # Doesn't have period and we want the period
        return "." + ext

    # Doesn't have period and we don't want the period
    return ext


def extract_ext(path: Union[pd.Series, str]) -> Union[pd.Series, str]:
    """
    Extract extensions from a file path.

    Parameters
    ----------
    path : Union[pd.Series, str]
        A series or a string containing a file path with an extension.

    Returns
    -------
    Union[pd.Series, str]
        A series or a string containing extracted extensions from the file path.
    """
    if isinstance(path, str):
        return os.path.splitext(path)[1]

    return path.apply(lambda p: os.path.splitext(p)[1])


def remove_ext(path: Union[pd.Series, str]) -> Union[pd.Series, str]:
    """
    Remove extensions from a file path.

    Parameters
    ----------
    path : Union[pd.Series, str]
        A series or a string containing a file path with an extension.

    Returns
    -------
    Union[pd.Series, str]
        A series or a string with the extension removed from the file path.
    """
    if isinstance(path, str):
        return os.path.splitext(path)[0]

    return path.apply(lambda p: os.path.splitext(p)[0])


def replace_ext(path: Union[pd.Series, str], replacement_ext: str) -> Union[pd.Series, str]:
    """
    Replace extensions in a file path with a new extension.

    Parameters
    ----------
    path : Union[pd.Series, str]
        A series or a string containing a file path with an extension.
    replacement_ext : str
        The new extension to replace the existing extension.

    Returns
    -------
    Union[pd.Series, str]
        A series or a string with the extension replaced by the new extension.
    """
    path_without_extension = remove_ext(path)

    if not replacement_ext.startswith("."):
        replacement_ext = "." + replacement_ext

    return path_without_extension + replacement_ext

def extract_filename(paths: pd.Series) -> pd.Series:
    """
    Extract filenames from a series of file paths.

    Parameters
    ----------
    paths : pandas.Series
        A series of file paths.

    Returns
    -------
    pd.Series
        A Series containing filenames extracted from the file paths.
    """
    return paths.str.split('/').str[-1]

def remove_common_segments(paths: pd.Series) -> pd.Series:
    """
    Removes the shared, or common, directory segments from a Series of paths.

    Parameters
    ----------
    paths : pd.Series
        A Series containing paths.

    Returns
    -------
    pd.Series
        Series with the common segments removed from each path.
    """
    common_prefix = os.path.commonprefix(list(paths[~paths.isna()].values))
    if common_prefix[-1] != "/":
        common_prefix = common_prefix[:common_prefix.rfind("/") + 1]

    return paths.str.slice(start=len(common_prefix))

def remove_common_prefix(paths: pd.Series) -> pd.Series:
    """
    Removes the shared, or common, prefix from a Series of paths.

    Parameters
    ----------
    paths : pd.Series
        A Series containing paths.

    Returns
    -------
    pd.Series
        Series with the common prefix removed from each path.
    """
    common_prefix = os.path.commonprefix(list(paths.values))
    return paths.str.slice(start=len(common_prefix))

def filenames_from_paths(
    paths: pd.Series,
    replacement_ext: Optional[str] = None,
    exclude_common_segments: bool = True,
    exclude_common_prefix: bool = False,
    warn_not_unique: bool = True,
):
    """
    Generate filenames from file paths.

    Useful for creating unique filenames from file paths, but this is not guaranteed,
    therefore a warning is issued by default.

    Parameters
    ----------
    paths : pd.Series
        A series containing file paths.
    replacement_ext : str, optional
        The new extension to replace existing extensions, no replacement by default.
    exclude_common_segments : bool, default True
        If True, removes the first directory segments shared by all of the files.
    exclude_common_prefix : bool, default False
        If True, excludes the common prefix from each filename to reduce filename lengths.
    warn_not_unique : bool, default True
        Whether to raise a warning if the resulting filenames are not unique.

    Returns
    -------
    pd.Series
        A series of generated filenames derived from the input file paths.

    Warnings
    --------
    Warn when warn_not_unique is True and the resulting filenames are not unique.
    """
    if exclude_common_segments and exclude_common_prefix:
        raise ValueError(
            "Cannot specify `exclude_common_segments` and `exclude_common_prefix`."
        )

    if replacement_ext is not None:
        # Replace extensions
        paths = replace_ext(paths, replacement_ext)

    if exclude_common_segments:
        paths = remove_common_segments(paths)

    if exclude_common_prefix:
        paths = remove_common_prefix(paths)

    # Convert from a path into a viable filename
    paths = paths.str.replace("/", "_").str.replace(" ", "_")

    if warn_not_unique and not paths.is_unique:
        warnings.warn("Resulting filenames are not unique.")

    return paths

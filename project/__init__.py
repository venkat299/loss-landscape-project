"""
Top-level package for the Loss Landscape Geometry project.

This package contains data generation utilities, model definitions,
training and experiment scripts, and loss landscape analysis tools.
"""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """
    Return the installed package version if available.

    This is safe to call even when the project is not installed
    as a package; in that case a default string is returned.

    Returns:
        str: Semantic version string or a fallback value.
    """
    try:
        return version("project")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]


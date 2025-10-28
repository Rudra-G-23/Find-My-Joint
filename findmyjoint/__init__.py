# findmyjoint/__init__.py
# This file makes 'findmyjoint' a Python package and exposes the public API.

import pandas as pd
from .core import FindMyJoint, profile_dataset, compare_columns
from .visualizer import FindMyJointVisual, build_pyvis_graph

__version__ = "0.0.1"

# --- One-liner wrapper functions ---

def compare(datasets: list, names: list = None, **kwargs) -> "pd.DataFrame":
    """
    A one-line function to compare columns across multiple DataFrames.

    Args:
        datasets: A list of pandas DataFrames.
        names: An optional list of names for the datasets.
        **kwargs: Additional arguments passed to `match_columns` (e.g., name_threshold).

    Returns:
        A pandas DataFrame with the comparison matrix.
    """
    fmj = FindMyJoint(datasets, names=names)
    return fmj.match_columns(**kwargs)

def network(datasets: list, names: list = None, out: str = "joint_graph.html", **kwargs):
    """
    A one-line function to generate and save an interactive relationship graph.
    
    Args:
        datasets: A list of pandas DataFrames.
        names: An optional list of names for the datasets.
        out: The output HTML file name.
        **kwargs: Additional arguments passed to `build_graph` (e.g., threshold).
    """
    fmj_visual = FindMyJointVisual(datasets, names=names)
    fmj_visual.build_graph(out=out, **kwargs)

# Expose the main functions and classes for easy import
__all__ = [
    'FindMyJoint',
    'FindMyJointVisual',
    'compare',
    'network',
    'profile_dataset',
    'compare_columns',
    'build_pyvis_graph'
]

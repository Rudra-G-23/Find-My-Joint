# findmyjoint.py
# A single-file implementation for detecting "joints" (matching columns) across datasets.

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import re
from itertools import combinations
import collections

# --- Core Helper Functions ---

def _normalize_name(name):
    """Converts a column name to a standardized format for comparison."""
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9\s]', ' ', name) # Remove punctuation
    name = re.sub(r'\s+', '_', name).strip('_') # Replace whitespace with underscore
    return name

def _get_coarse_dtype(dtype):
    """Maps a pandas dtype to a coarser category (numeric, string, datetime, other)."""
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numeric'
    if pd.api.types.is_string_dtype(dtype) or dtype == 'object':
        return 'string'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'datetime'
    return 'other'

def profile_dataset(df: pd.DataFrame, name: str, sample_frac: float = 0.1, max_sample_size: int = 1000) -> pd.DataFrame:
    """
    Analyzes a DataFrame and returns a metadata DataFrame describing each column.

    Args:
        df: The input DataFrame to profile.
        name: The name to assign to this dataset.
        sample_frac: The fraction of the dataset to use for content sampling.
        max_sample_size: The maximum number of unique values to store in the sample.

    Returns:
        A pandas DataFrame where each row represents a column from the input df.
    """
    meta_records = []
    num_rows = len(df)
    if num_rows == 0:
        return pd.DataFrame()

    for col_name in df.columns:
        col = df[col_name]
        
        # Take a sample for content analysis to avoid loading all unique values into memory
        sample_size = min(max(1, int(num_rows * sample_frac)), num_rows)
        col_sample = col.sample(n=sample_size, random_state=1) if num_rows > 0 else col
        
        unique_values = col_sample.dropna().unique()
        
        # Further cap the number of unique values stored
        if len(unique_values) > max_sample_size:
            unique_values = np.random.choice(unique_values, max_sample_size, replace=False)

        record = {
            'dataset': name,
            'column': col_name,
            'dtype': str(col.dtype),
            'coarse_dtype': _get_coarse_dtype(col.dtype),
            'n_rows': num_rows,
            'n_unique': col.nunique(),
            'null_pct': col.isnull().mean(),
            'sample_values': set(unique_values),
            'normalized_name': _normalize_name(col_name)
        }
        meta_records.append(record)

    return pd.DataFrame(meta_records)

def compare_columns(meta: pd.DataFrame, name_threshold: float = 0.7, content_threshold: float = 0.5) -> pd.DataFrame:
    """
    Pairwise compares columns based on metadata to find potential joins.

    Args:
        meta: A metadata DataFrame created by `profile_dataset`.
        name_threshold: Minimum name similarity to be considered a match.
        content_threshold: Minimum content similarity to be considered a match.

    Returns:
        A pandas DataFrame detailing the comparison between each potential column pair.
    """
    comparison_records = []
    
    # Generate unique pairs of columns from different datasets
    for (idx1, row1), (idx2, row2) in combinations(meta.iterrows(), 2):
        if row1['dataset'] == row2['dataset']:
            continue

        # 1. Name Similarity
        name_sim = fuzz.token_sort_ratio(row1['normalized_name'], row2['normalized_name']) / 100.0
        
        # 2. Dtype Match
        dtype_match = 1.0 if row1['coarse_dtype'] == row2['coarse_dtype'] else 0.0
        
        # 3. Content Similarity (Jaccard)
        set1 = row1['sample_values']
        set2 = row2['sample_values']
        intersection_len = len(set1.intersection(set2))
        union_len = len(set1.union(set2))
        content_sim = intersection_len / union_len if union_len > 0 else 0.0

        # Heuristic filtering: skip pairs with very low similarity early
        if name_sim < name_threshold * 0.8 and content_sim < content_threshold * 0.8:
            continue
            
        # 4. Suggestion Logic
        suggestion = "Review"
        if name_sim >= 0.9 and dtype_match == 1.0 and content_sim >= 0.9:
            suggestion = "Merge-safe"
        elif name_sim >= 0.8 and dtype_match == 1.0 and content_sim >= 0.7:
            suggestion = "High-confidence join"
        elif name_sim < 0.6 and content_sim > 0.8 and dtype_match == 1.0:
            suggestion = "Rename candidate (content matches)"
        elif name_sim > 0.8 and dtype_match == 0.0 and content_sim > 0.7:
            suggestion = "Cast type (name/content match)"
        elif name_sim > 0.8 and dtype_match == 1.0 and content_sim < 0.3:
            suggestion = "Check content mismatch"

        # 5. Combined Edge Weight
        # Weights: name=0.4, dtype=0.2, content=0.4
        edge_weight = (0.4 * name_sim) + (0.2 * dtype_match) + (0.4 * content_sim)

        record = {
            'dataset_left': row1['dataset'],
            'column_left': row1['column'],
            'dataset_right': row2['dataset'],
            'column_right': row2['column'],
            'name_sim': name_sim,
            'dtype_match': dtype_match,
            'content_sim': content_sim,
            'suggestion': suggestion,
            'edge_weight': edge_weight
        }
        comparison_records.append(record)
    
    if not comparison_records:
        return pd.DataFrame(columns=['dataset_left', 'column_left', 'dataset_right', 'column_right', 'name_sim', 'dtype_match', 'content_sim', 'suggestion', 'edge_weight'])

    results_df = pd.DataFrame(comparison_records)
    return results_df.sort_values(by='edge_weight', ascending=False).reset_index(drop=True)


# --- Public API Class and Wrappers ---

class FindMyJoint:
    """A class to manage profiling and matching columns across multiple datasets."""
    
    def __init__(self, datasets: list, names: list = None, sample_frac: float = 0.1):
        """
        Args:
            datasets: A list of pandas DataFrames.
            names: An optional list of names for the datasets.
            sample_frac: The fraction of each dataset to sample for content analysis.
        """
        if not all(isinstance(df, pd.DataFrame) for df in datasets):
            raise TypeError("All items in 'datasets' must be pandas DataFrames.")
        if names and len(datasets) != len(names):
            raise ValueError("The number of datasets must match the number of names.")
            
        self.datasets = datasets
        self.names = names or [f"ds{i}" for i in range(len(datasets))]
        self.sample_frac = sample_frac
        self.meta = None
        self.matches = None

    def profile(self) -> pd.DataFrame:
        """Profiles all datasets and stores the combined metadata."""
        frames = []
        for df, name in zip(self.datasets, self.names):
            frames.append(profile_dataset(df, name, sample_frac=self.sample_frac))
        self.meta = pd.concat(frames, ignore_index=True)
        return self.meta

    def match_columns(self, name_threshold: float = 0.7, content_threshold: float = 0.5) -> pd.DataFrame:
        """
        Finds and scores potential join columns across all datasets.

        Args:
            name_threshold: Minimum name similarity (0-1) to consider.
            content_threshold: Minimum content similarity (0-1) to consider.
        
        Returns:
            A DataFrame of column pair comparisons and join suggestions.
        """
        if self.meta is None:
            print("Metadata not found. Profiling datasets first...")
            self.profile()
            
        print("Comparing columns across datasets...")
        self.matches = compare_columns(
            self.meta,
            name_threshold=name_threshold,
            content_threshold=content_threshold
        )
        return self.matches

# --- One-liner wrapper functions ---

def compare(datasets: list, names: list = None, **kwargs) -> pd.DataFrame:
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

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running findmyjoint Example ---")

    # 1. Create small toy datasets
    df1 = pd.DataFrame({
        'age': [21, 25, 30, 45],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'user_id': ['001', '002', '003', '004']
    })

    df2 = pd.DataFrame({
        'Age': ['21', '25', '30', '45'],
        'full_name': ['Alice', 'Bob', 'Charlie', 'David'],
        'customer_id': [1, 2, 3, 4],
        'country': ['USA', 'CAN', 'USA', 'MEX']
    })
    
    df3 = pd.DataFrame({
        'client_identifier': ['001', '002', '003', '004'],
        'location': ['USA', 'CAN', 'USA', 'MEX'],
        'years_old': [21, 25, 30, 45]
    })


    # 2. Use the one-line `compare` function
    print("\n[INFO] Using the one-line `compare` function on df1 and df2...")
    matrix = compare([df1, df2], names=['hr_system', 'crm_data'])
    
    # Display the top results
    print("Top 5 comparison results for df1 vs df2:")
    print(matrix.head())

    print("\n" + "="*50 + "\n")

    # 3. Use the class interface for more control with three datasets
    print("[INFO] Using the `FindMyJoint` class for a more complex comparison (3 datasets)...")
    fmj_instance = FindMyJoint(datasets=[df1, df2, df3], names=['hr', 'crm', 'finance'])
    
    # You can inspect the metadata
    # metadata = fmj_instance.profile()
    # print("\nSample of generated metadata:")
    # print(metadata.head())
    
    # Find all matches
    all_matches = fmj_instance.match_columns(name_threshold=0.6)

    print("\nTop 10 comparison results across all 3 datasets:")
    print(all_matches.head(10))
    
    print("\n--- Example Finished ---")

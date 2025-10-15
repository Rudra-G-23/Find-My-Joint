# findmyjoint_visualizer.py
# A single-file implementation for detecting and visualizing "joints" (matching columns) across datasets.

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import re
from itertools import combinations
import webbrowser
import os
from pyvis.network import Network

# --- Core Helper Functions ---

def _normalize_name(name: str) -> str:
    """Converts a column name to a standardized format for comparison."""
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9\s]', ' ', name) # Remove punctuation
    name = re.sub(r'\s+', '_', name).strip('_') # Replace whitespace with underscore
    return name

def _get_coarse_dtype(dtype) -> str:
    """Maps a pandas dtype to a coarser category (numeric, string, datetime, other)."""
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numeric'
    if pd.api.types.is_string_dtype(dtype) or dtype == 'object':
        return 'string'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'datetime'
    return 'other'

# --- Profiling and Matching Logic (from previous step) ---

def profile_dataset(df: pd.DataFrame, name: str, sample_frac: float = 0.1, max_sample_size: int = 1000) -> pd.DataFrame:
    """Analyzes a DataFrame and returns a metadata DataFrame describing each column."""
    meta_records = []
    num_rows = len(df)
    if num_rows == 0:
        return pd.DataFrame()

    for col_name in df.columns:
        col = df[col_name]
        sample_size = min(max(1, int(num_rows * sample_frac)), num_rows)
        col_sample = col.sample(n=sample_size, random_state=1) if num_rows > 0 else col
        unique_values = col_sample.dropna().unique()
        
        if len(unique_values) > max_sample_size:
            unique_values = np.random.choice(unique_values, max_sample_size, replace=False)

        record = {
            'dataset': name, 'column': col_name, 'dtype': str(col.dtype),
            'coarse_dtype': _get_coarse_dtype(col.dtype), 'n_rows': num_rows,
            'n_unique': col.nunique(), 'null_pct': col.isnull().mean(),
            'sample_values': set(unique_values), 'normalized_name': _normalize_name(col_name)
        }
        meta_records.append(record)
    return pd.DataFrame(meta_records)

def compare_columns(meta: pd.DataFrame, name_threshold: float = 0.7, content_threshold: float = 0.5) -> pd.DataFrame:
    """Pairwise compares columns based on metadata to find potential joins."""
    comparison_records = []
    
    for (idx1, row1), (idx2, row2) in combinations(meta.iterrows(), 2):
        if row1['dataset'] == row2['dataset']:
            continue

        name_sim = fuzz.token_sort_ratio(row1['normalized_name'], row2['normalized_name']) / 100.0
        dtype_match = 1.0 if row1['coarse_dtype'] == row2['coarse_dtype'] else 0.0
        
        set1, set2 = row1['sample_values'], row2['sample_values']
        intersection_len = len(set1.intersection(set2))
        union_len = len(set1.union(set2))
        content_sim = intersection_len / union_len if union_len > 0 else 0.0

        if name_sim < name_threshold * 0.8 and content_sim < content_threshold * 0.8:
            continue
            
        suggestion = "Review"
        if name_sim >= 0.9 and dtype_match == 1.0 and content_sim >= 0.9: suggestion = "Merge-safe"
        elif name_sim >= 0.8 and dtype_match == 1.0 and content_sim >= 0.7: suggestion = "High-confidence join"
        elif name_sim < 0.6 and content_sim > 0.8 and dtype_match == 1.0: suggestion = "Rename candidate (content matches)"
        elif name_sim > 0.8 and dtype_match == 0.0 and content_sim > 0.7: suggestion = "Cast type (name/content match)"
        elif name_sim > 0.8 and dtype_match == 1.0 and content_sim < 0.3: suggestion = "Check content mismatch"

        edge_weight = (0.4 * name_sim) + (0.2 * dtype_match) + (0.4 * content_sim)

        record = {
            'dataset_left': row1['dataset'], 'column_left': row1['column'],
            'dataset_right': row2['dataset'], 'column_right': row2['column'],
            'name_sim': name_sim, 'dtype_match': dtype_match, 'content_sim': content_sim,
            'suggestion': suggestion, 'edge_weight': edge_weight
        }
        comparison_records.append(record)
    
    if not comparison_records:
        return pd.DataFrame(columns=list(record.keys()))

    return pd.DataFrame(comparison_records).sort_values(by='edge_weight', ascending=False).reset_index(drop=True)

# --- NEW Visualization Logic ---

def build_pyvis_graph(matches_df: pd.DataFrame, meta_df: pd.DataFrame, threshold: float = 0.5) -> Network:
    """
    Builds an interactive pyvis network graph from the comparison results.

    Args:
        matches_df: The DataFrame of column comparisons.
        meta_df: The DataFrame of column metadata.
        threshold: The minimum edge_weight to include in the graph.

    Returns:
        A pyvis Network object.
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.force_atlas_2based(gravity=-60, central_gravity=0.01, spring_length=200, spring_strength=0.08)

    # Add nodes (each column is a node, grouped by dataset)
    for _, row in meta_df.iterrows():
        node_id = f"{row['dataset']}:{row['column']}"
        net.add_node(node_id, label=row['column'], group=row['dataset'], title=f"Dataset: {row['dataset']}\nColumn: {row['column']}\nDtype: {row['dtype']}")
        
    # Add edges for connections meeting the threshold
    filtered_matches = matches_df[matches_df['edge_weight'] >= threshold]
    for _, row in filtered_matches.iterrows():
        source = f"{row['dataset_left']}:{row['column_left']}"
        target = f"{row['dataset_right']}:{row['column_right']}"
        
        # Define edge color based on suggestion for quick visual feedback
        color_map = {
            "Merge-safe": "#2ECC71", # Green
            "High-confidence join": "#3498DB", # Blue
            "Rename candidate (content matches)": "#F1C40F", # Yellow
            "Cast type (name/content match)": "#E67E22", # Orange
            "Check content mismatch": "#E74C3C", # Red
            "Review": "#95A5A6" # Gray
        }
        
        title = (f"Suggestion: {row['suggestion']}\n"
                 f"--------------------------\n"
                 f"Edge Weight: {row['edge_weight']:.2f}\n"
                 f"Name Sim: {row['name_sim']:.2f}\n"
                 f"Content Sim: {row['content_sim']:.2f}\n"
                 f"Dtype Match: {row['dtype_match']:.0f}")

        net.add_edge(source, target, value=row['edge_weight'] * 10, title=title, color=color_map.get(row['suggestion'], "grey"))

    net.show_buttons(filter_=['physics'])
    return net


# --- Public API Class and Wrappers ---

class FindMyJoint:
    """A class to manage profiling, matching, and visualizing column relationships."""
    
    def __init__(self, datasets: list, names: list = None, sample_frac: float = 0.1):
        if not all(isinstance(df, pd.DataFrame) for df in datasets):
            raise TypeError("All items in 'datasets' must be pandas DataFrames.")
        if names and len(datasets) != len(names):
            raise ValueError("The number of datasets must match the number of names.")
            
        self.datasets = datasets
        self.names = names or [f"ds{i}" for i in range(len(datasets))]
        self.sample_frac = sample_frac
        self.meta = None
        self.matches = None
        self.graph = None

    def profile(self) -> pd.DataFrame:
        """Profiles all datasets and stores the combined metadata."""
        frames = [profile_dataset(df, name, self.sample_frac) for df, name in zip(self.datasets, self.names)]
        self.meta = pd.concat(frames, ignore_index=True)
        return self.meta

    def match_columns(self, name_threshold: float = 0.7, content_threshold: float = 0.5) -> pd.DataFrame:
        """Finds and scores potential join columns across all datasets."""
        if self.meta is None: self.profile()
        self.matches = compare_columns(self.meta, name_threshold, content_threshold)
        return self.matches

    def build_graph(self, out: str = "joint_graph.html", threshold: float = 0.5, open_in_browser: bool = True):
        """Builds and saves an interactive network graph of column relationships."""
        if self.matches is None: self.match_columns()
        
        print(f"Building graph with connections stronger than {threshold}...")
        self.graph = build_pyvis_graph(self.matches, self.meta, threshold=threshold)
        
        try:
            self.graph.show(out)
            print(f"Success! Interactive graph saved to: {os.path.abspath(out)}")
            if open_in_browser:
                print("Opening in your default browser...")
                webbrowser.open(f"file://{os.path.abspath(out)}")
        except Exception as e:
            print(f"Error saving or opening graph: {e}")

# --- One-liner wrapper functions ---

def compare(datasets: list, names: list = None, **kwargs) -> pd.DataFrame:
    """A one-line function to compare columns across multiple DataFrames."""
    fmj = FindMyJoint(datasets, names=names)
    return fmj.match_columns(**kwargs)

def network(datasets: list, names: list = None, out: str = "joint_graph.html", **kwargs):
    """A one-line function to generate and save an interactive relationship graph."""
    fmj = FindMyJoint(datasets, names=names)
    fmj.build_graph(out=out, **kwargs)

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running findmyjoint Example with Visualization ---")

    # 1. Create toy datasets
    df1 = pd.DataFrame({'age': [21, 25, 30, 45], 'name': ['Alice', 'Bob', 'Charlie', 'David'], 'user_id': ['001', '002', '003', '004']})
    df2 = pd.DataFrame({'Age': ['21', '25', '30', '45'], 'full_name': ['Alice', 'Bob', 'Charlie', 'David'], 'customer_id': [1, 2, 3, 4]})
    df3 = pd.DataFrame({'client_identifier': ['001', '002', '003', '004'], 'location': ['USA', 'CAN', 'USA', 'MEX'], 'years_old': [21, 25, 30, 45]})

    # 2. Use the one-line `compare` function to see the data
    print("\n[STEP 1] Generating the comparison matrix...")
    matrix = compare([df1, df2, df3], names=['hr', 'crm', 'finance'])
    print("Top 5 comparison results:")
    print(matrix.head())

    # 3. Use the one-line `network` function to generate the graph
    print("\n[STEP 2] Generating the interactive network graph...")
    # This will create and automatically open 'joint_graph.html'
    network([df1, df2, df3], names=['hr', 'crm', 'finance'], threshold=0.6)
    
    print("\n--- Example Finished ---")



"""
 ### How to Run This Code

1.  **Save:** Save the code above into a file named `findmyjoint_visualizer.py`.

2.  **Install Dependencies:** You'll need `pandas`, `rapidfuzz`, and `pyvis`. You can install them with pip:
    ```bash
    pip install pandas rapidfuzz pyvis
    ```

3.  **Execute:** Run the script from your terminal:
    ```bash
    python findmyjoint_visualizer.py

"""   

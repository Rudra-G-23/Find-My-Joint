# findmyjoint/visualizer.py
# Extends the core logic with pyvis-based network visualization.

import webbrowser
import os
import pandas as pd
from pyvis.network import Network
from .core import FindMyJoint, profile_dataset, compare_columns

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


class FindMyJointVisual(FindMyJoint):
    """
    Extends FindMyJoint to add graph visualization capabilities.
    """
    
    def build_graph(self, out: str = "joint_graph.html", threshold: float = 0.5, open_in_browser: bool = True):
        """Builds and saves an interactive network graph of column relationships."""
        if self.matches is None:
            print("No matches found. Running match_columns() first...")
            self.match_columns()
        
        if self.matches.empty:
            print("No column matches found meeting criteria. Graph will be empty.")
            return

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

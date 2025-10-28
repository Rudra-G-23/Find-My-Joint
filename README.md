# FindMyJoint

A Python utility to analyze and compare columns across multiple pandas DataFrames, suggesting potential join keys and visualizing the relationships.

When working with multiple disparate datasets, finding common columns to join them on is a tedious manual task. findmyjoint automates this by:

1. Profiling each DataFrame's columns (dtype, uniqueness, nulls).
2. Comparing all possible column pairs across datasets.
3. Scoring pairs based on name similarity (using rapidfuzz) and content similarity (using Jaccard index).
4. Suggesting join confidence levels.
5. Visualizing the connections as an interactive network graph (using pyvis).

## Installation
You will be able to install this via pip once it's published:


```bash
pip install findmyjoint
```


## Quickstart

You can get a comparison matrix or an interactive graph with a single line of code.

### 1. Create toy datasets
```bash
df1 = pd.DataFrame({
    'age': [21, 25, 30, 45],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'user_id': ['001', '002', '003', '004']
})

df2 = pd.DataFrame({
    'Age': ['21', '25', '30', '45'],
    'full_name': ['Alice', 'Bob', 'Charlie', 'David'],
    'customer_id': [1, 2, 3, 4]
})

df3 = pd.DataFrame({
    'client_identifier': ['001', '002', '003', '004'],
    'location': ['USA', 'CAN', 'USA', 'MEX'],
    'years_old': [21, 25, 30, 45]
})

datasets = [df1, df2, df3]
names = ['hr', 'crm', 'finance']

# 2. Get the comparison matrix
print("--- Comparison Matrix ---")
matrix = fmj.compare(datasets, names=names, name_threshold=0.6)
print(matrix.head())

# 3. Generate the interactive network graph
print("\n--- Generating Network Graph ---")

# This will create and automatically open 'joint_graph.html'
fmj.network(datasets, names=names, threshold=0.6)
print("Graph 'joint_graph.html' created.")
```
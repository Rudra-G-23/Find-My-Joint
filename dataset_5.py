import pandas as pd
import numpy as np

# Create mock datasets
np.random.seed(42)

# Dataset 1
df1 = pd.DataFrame({
    'ID': np.arange(1, 11),
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack'],
    'Age': np.random.randint(20, 50, size=10),
    'Salary': np.random.randint(30000, 100000, size=10),
    'City': ['NY', 'LA', 'SF', 'Miami', 'Chicago', 'LA', 'NY', 'SF', 'Miami', 'Chicago']
})

# Dataset 2 (similar to dataset 1 but with slight variations)
df2 = df1.copy()
df2['age'] = df2['Age'] + 2  # Slight variation in 'Age'
df2['location city'] = ['NY', 'LA', 'Chicago', 'SF', 'Miami', 'LA', 'NY', 'SF', 'Chicago', 'Miami']  # Slight change in 'City'

# Dataset 3 (totally different structure)
df3 = pd.DataFrame({
    'UserID': np.arange(1, 11),
    'FullName': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack'],
    'YearsExperience': np.random.randint(1, 20, size=10),
    'AnnualIncome': np.random.randint(30000, 120000, size=10),
    'Location': ['NY', 'LA', 'SF', 'Miami', 'Chicago', 'LA', 'NY', 'SF', 'Miami', 'Chicago']
})

# Dataset 4 (similar to dataset 1 but with a new column)
df4 = df1.copy()
df4['Department'] = ['HR', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR']  # New column

# Dataset 5 (totally different data)
df5 = pd.DataFrame({
    'ItemID': np.arange(1, 11),
    'ProductName': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Headphones', 'Speaker', 'Camera', 'Watch'],
    'Price': np.random.randint(100, 2000, size=10),
    'Stock': np.random.randint(0, 100, size=10),
    'Category': ['Electronics'] * 10
})

# :material-star-remove-outline: Quick Start

## 1. Create a Datasets

!!! abstract

    ```py title="Install Libraries"
    pip install findmyjoint pandas -q
    ```
    ```py title="Create Data Frame"
    # Import
    import pandas as pd
    import findmyjoint as fmj
    
    # Create data frames
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
    ```

## 2. Potential Join Data Frame
!!! tip
    === "Input"
        ```py title="Your Joint Requirement"
        datasets = [df1, df2, df3]
        names = ['hr', 'crm', 'finance']
        ```

        ```py title="Comparison Matrix"
        print("--- Comparison Matrix ---")
        matrix = fmj.compare(datasets, names=names, name_threshold=0.6)
        print(matrix.head())
        ```
    === "Output DataFrame"
        
        | dataset_left | column_left | dataset_right | column_right      | name_sim | dtype_match | content_sim | suggestion                         | edge_weight |
        | ------------ | ----------- | ------------- | ----------------- | -------- | ----------- | ----------- | ---------------------------------- | ----------- |
        | hr           | name        | crm           | full_name         | 0.615385 | 1.0         | 1.0         | Review                             | 0.846154    |
        | hr           | user_id     | finance       | client_identifier | 0.333333 | 1.0         | 1.0         | Rename candidate (content matches) | 0.733333    |
        | hr           | age         | finance       | years_old         | 0.166667 | 1.0         | 1.0         | Rename candidate (content matches) | 0.666667    |
        | hr           | name        | crm           | Age               | 0.571429 | 1.0         | 0.0         | Review                             | 0.428571    |
        | hr           | age         | crm           | Age               | 1.000000 | 0.0         | 0.0         | Review                             | 0.400000    |
        | hr           | user_id     | crm           | customer_id       | 0.777778 | 0.0         | 0.0         | Review                             | 0.311111    |
        | hr           | user_id     | finance       | years_old         | 0.500000 | 0.0         | 0.0         | Review                             | 0.200000    |

## 3. Joint Visualization
!!! success "Interactive Network Graph"
    ```python
    # This will create and automatically open 'joint_graph.html'

    print("\n--- Generating Network Graph ---")
    fmj.network(datasets, names=names, threshold=0.6)
    print("Graph 'joint_graph.html' created.")
    ```

## 4. Output Graph
<iframe width="560" height="315" src="https://www.youtube.com/embed/Bf8W8jgtW8Y?si=sJamHgLZNz5ArPfc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
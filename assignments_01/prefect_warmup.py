import pandas as pd
import numpy as np
from prefect import task, flow

@task
def create_series(arr):
    values = pd.Series(arr)
    return values

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    series_info = {
        "mean": np.mean(series),
        "median": np.median(series),
        "std": np.std(series),
        "mode": series.mode()[0]
    }

    return series_info

@flow
def data_pipeline():
    arr = np.array([
        12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
        18.0, 14.0, 16.0, 22.0, np.nan, 13.0
    ])

    values = create_series(arr)
    cleaned = clean_data(values)
    summary = summarize_data(cleaned)
    return summary


if __name__ == '__main__':
    data_pipeline()


"""
Note on running this in the command line:

1. prefect server start
2. python hello_prefect.py
"""

# ----COMMENT BLOCK----
"""
Q1) Why might Prefect be overkill for this example?

For a small program like this, Prefect can be overkill because there are
only a few simple tasks and no complicated workflow. Prefect adds extra
dependencies, logging, task management, and overhead that are not really
needed for such a small program. It would be simpler to run these functions
directly with normal Python.


Q2) When would Prefect be useful?

Prefect would be useful for a larger data pipeline where there are many
different tasks and data sources. For example, in the sports industry, a
team could have different tasks for collecting data from multiple sources,
cleaning and transforming the data, combining it into one dataset, loading
it into a database, and then creating reports or visualizations. Prefect
would be useful for organizing those tasks, tracking their progress, and
handling failures or retries.
"""

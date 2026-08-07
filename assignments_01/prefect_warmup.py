import pandas as pd
import numpy as np
from prefect import task, flow

@task
def create_series(arr):
    return pd.Series(arr)

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
def data_pipeline(arr):
    values = create_series(arr)
    cleaned = clean_data(values)
    summary = summarize_data(cleaned)
    return summary


if __name__ == "__main__":
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    print(data_pipeline(arr))
    #{'mean': 14.88888888888889, 'median': 14.0, 'std': 3.314763086705844, 'mode': 14.0}

"""
Note on running this in the command line:
1) prefect server start
2) prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api (in another window)
3) python hello_prefect.py
"""

# ----COMMENT BLOCK----
"""
Q1) For something small like this, the extra information and potential logging if we were to do that
might make the program load or run slower and isn't fully needed in this case with the extra dependencies 

Q2) I think a realist scenario would be in the sports industry. I can imagine a team would have a lot of different
places/sources to pull data from so that can be its own task (with more subtasks with the different sources and then
combining the data into one table for example). Then with each of those sources we need to do some cleaning, 
transformations, standardizations, etc... and then we can load it into a database to do some queries and analysis and 
plotting. 
"""

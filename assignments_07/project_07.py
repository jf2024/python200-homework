import os
import glob
import pandas as pd
from scipy.stats import pearsonr
from smolagents import tool
from smolagents import CodeAgent, OpenAIServerModel
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"
df = None

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")

# Task 1
df = None

# help from dominic on the slack regarding the agent not being able to access the df
@tool
def get_happiness_dataframe() -> pd.DataFrame:
    """Return the loaded World Happiness DataFrame for custom analysis and plotting.

    Use this tool when you need the full World Happiness dataset to perform
    calculations or create custom plots that cannot be completed with the
    summary, correlation, or ranking tools.

    Returns:
        pd.DataFrame: A copy of the loaded World Happiness dataset. If data
        has not been loaded yet, this tool loads it first.
    """

    global df

    if df is None:
        load_happiness_data()

    return df.copy()

@tool
def load_happiness_data() -> dict:
    """Load the happiness dataset."""

    global df

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
    }

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Use this tool when you need a statistical summary of one column from the
    World Happiness dataset. The column must exist in the dataset, and the
    dataset must have been loaded with ``load_happiness_data`` first.

    Args:
        column: The exact name of the column for which descriptive statistics
            should be calculated.

    Returns:
        dict: The result of pandas ``describe().to_dict()`` for the requested
            column. If the dataset has not been loaded or the column does not
            exist, returns a dictionary containing an ``error`` message.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found in the dataset."}

    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Use this tool to measure the linear relationship between two numeric
    columns in the loaded World Happiness dataset. Both columns must exist,
    contain numeric data, and have enough valid paired observations for a
    Pearson correlation to be calculated.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        dict: A dictionary containing:
            - ``col1``: The name of the first column.
            - ``col2``: The name of the second column.
            - ``pearson_r``: The Pearson correlation coefficient rounded to
              four decimal places.
            - ``p_value``: The Pearson correlation p-value rounded to four
              decimal places.

        If the dataset is not loaded, either column is missing, the columns
        are not numeric, or the correlation cannot be calculated, returns a
        dictionary containing an ``error`` message.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' was not found in the dataset."}

    if col2 not in df.columns:
        return {"error": f"Column '{col2}' was not found in the dataset."}

    if not pd.api.types.is_numeric_dtype(df[col1]):
        return {"error": f"Column '{col1}' must be numeric."}

    if not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": f"Column '{col2}' must be numeric."}

    valid_data = df[[col1, col2]].dropna()

    if len(valid_data) < 2:
        return {
            "error": "Not enough valid paired observations to compute correlation."
        }

    try:
        pearson_r, p_value = pearsonr(
            valid_data[col1],
            valid_data[col2],
        )
    except Exception as exc:
        return {"error": f"Could not compute correlation: {exc}"}

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(pearson_r, 4),
        "p_value": round(p_value, 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Use this tool when you need to identify the countries with the highest
    values for a particular metric in the World Happiness dataset for a
    specified year. The dataset must first be loaded with
    ``load_happiness_data``.

    Args:
        column: The name of the column used to rank countries. The column
            must exist in the dataset and should contain numeric values.
        year: The year for which countries should be ranked.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        dict: A dictionary containing:
            - ``year``: The requested year.
            - ``column``: The metric used for ranking.
            - ``results``: A list of dictionaries, where each dictionary
              contains ``country`` and the requested column's value.

        If the dataset is not loaded, the requested column is missing, the
        ``year`` column is unavailable, or the input is otherwise invalid,
        returns a dictionary containing an ``error`` message.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if "year" not in df.columns:
        return {"error": "The dataset does not contain a 'year' column."}

    if "Country" not in df.columns:
        return {"error": "The dataset does not contain a 'Country' column."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found in the dataset."}

    if not isinstance(year, int):
        return {"error": "Year must be an integer."}

    if not isinstance(n, int) or n <= 0:
        return {"error": "n must be a positive integer."}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' must be numeric."}

    year_data = df[df["year"] == year]

    if year_data.empty:
        return {"error": f"No data found for year {year}."}

    year_data = year_data.dropna(subset=[column, "Country"])

    top_rows = year_data.sort_values(
        by=column,
        ascending=False,
    ).head(n)

    results = [
        {
            "country": row["Country"],
            column: row[column],
        }
        for _, row in top_rows.iterrows()
    ]

    return {
        "year": year,
        "column": column,
        "results": results,
    }

# Task 2
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries, get_happiness_dataframe],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# Task 3
queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


# Task 4
my_query_1 = (
    "For 2020, what are the top 10 countries by Happiness score? "
    "Explain the result briefly."
)
# This query triggered tool use by using the get_top_n_countries tool to retrieve
    # the top 10 countries and their Happiness scores for 2020

my_query_2 = (
    "Using the happiness dataset, write Python code to calculate the average "
    "Happiness score for each year and create a line plot showing how the "
    "average Happiness score changed over time. Save the plot as "
    "'outputs/average_happiness_by_year.png'."
)
# This query trigged both tool use and code generation. The agent first used the
    # get_happiness_dataframe tool to access the dataset, then generated and executed python
    # code using pandas and matplotlib to calculate the yearly averages and making the plot.

if __name__ == "__main__":

    for query in queries:
        print(f"query for task3: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    print(f"query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)


    print(f"query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)

# Task 5

# 1. The agent reported a correlation of 0.6313 and a p-value of 0.0, and
# described the correlation as "statistically significant." It appears to have used the general rule of thumb
# of p < 0.05. Since the p-value was below 0.05, it concludes that there is a statsitical signficant. 

#2. I was surprised that the agent was able to generate and execute Python code to
# create the requested plot and save it to the specified output path. And overall the avg happienss by year
# looks pretty decent but the happiness by region looks pretty scuffed which shows that the model can't do it
# perfrectly on its own and its not always reliable.
#

# 3. I think something useful would be an inspection/review tool. It could
    # open a generated plot and analyze whether the chart was created correctly/faithfully. Things like
    # title, legend, axes, is there too much data on the graph that its unreadable? (hence the happiness by region).
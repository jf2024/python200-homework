from prefect import task, flow
from prefect.logging import get_run_logger 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()

    directory = '../data/happiness_project/'
    all_data = []

    for file in os.listdir(directory):
        year = file.split("_")[-1].split(".")[0]

        if file.endswith(".csv"):
            full_path = directory + file
            df = pd.read_csv(full_path, sep=';', decimal=',')
            df['year'] = year
            if year == 2024:
                df = df.rename(columns={'Ladder score': 'Happiness score'})
            all_data.append(df)

    merged_df = pd.concat(all_data)
    merged_df.to_csv('outputs/merged_happiness.csv', index=False)
    logger.info("Happiness files merged successfully and inside of outputs folder")


@flow
def happiness_pipeline():
    load_data()


if __name__ == "__main__":
    happiness_pipeline()
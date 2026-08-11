from prefect import task, flow
from prefect.logging import get_run_logger 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()

    directory = '../data/happiness_project/'
    all_data = []

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            year = file.split("_")[-1].split(".")[0]
            full_path = directory + file
            df = pd.read_csv(full_path, sep=';', decimal=',')
            df['year'] = year
            if year == '2024':
                df = df.rename(columns={'Ladder score': 'Happiness score'})
            all_data.append(df)

    merged_df = pd.concat(all_data)
    merged_df.to_csv('outputs/merged_happiness.csv', index=False)
    logger.info("Happiness files merged successfully and inside of outputs folder")
    return merged_df

@task
def descriptive_stats(dataframe):
    logger = get_run_logger()

    happiness_stats = {
        "mean": dataframe['Happiness score'].mean(),
        "median": dataframe['Happiness score'].median(),
        "std": dataframe['Happiness score'].std(),
    }
    logger.info("Mean, median, and standard deviation for happiness score: \n%s", happiness_stats)

    regional_stats = dataframe.groupby(['year', 'Regional indicator'])['Happiness score'].mean()
    logger.info("Average Happiness Score by year and region:\n%s", regional_stats)

@task
def visuals(dataframe):
    logger = get_run_logger()

    # histogram 
    plt.hist(dataframe['Happiness score'], bins=25, color="purple", edgecolor="black")
    plt.title("Happiness Score between 2015-2024")
    plt.xlabel("Happiness Score Range")
    plt.ylabel("Counts of scores")
    plt.savefig('outputs/happiness_histogram.png')
    plt.show()
    logger.info('Histogram of happiness score completed, saved as "happiness_histogram" in outputs folder')

    # boxplot
    dataframe.boxplot(by='year', column='Happiness score')
    plt.title("Happiness Scores by year")
    plt.suptitle("") 
    plt.ylabel("Happiness Scores")
    plt.xlabel('Year')
    plt.savefig('outputs/happiness_by_year.png')
    plt.show()
    logger.info('Boxplot of happiness score by year done, saved as "happiness_by_year" in outputs folder')

    # scatterplot
    plt.scatter(dataframe['GDP per capita'], dataframe['Happiness score'], color="green")
    plt.title("GDP per capita vs Happiness Score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")
    plt.savefig('outputs/gdp_vs_happiness.png')
    plt.show()
    logger.info('Scatterplot of gdp and happiness score, saved as "gdp_vs_happiness" in outputs folder')

    # correlation heatmap
    corr = dataframe.select_dtypes((int, float)).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig('outputs/correlation_heatmap.png')
    plt.show()
    logger.info('Correlation heatmap inside of outputs folder')

@task
def hypo_test(dataframe):
    logger = get_run_logger()

    # first test
    happiness_2019 = dataframe[dataframe['year'] == '2019']['Happiness score']
    happiness_2020 = dataframe[dataframe['year'] == '2020']['Happiness score']
    mean_2019 = happiness_2019.mean()
    mean_2020 = happiness_2020.mean()
    t_stat1, p_val1 = stats.ttest_ind(happiness_2019, happiness_2020)
    logger.info("Mean happiness for 2019 and 2020 respectfully: %.2f, %.2f", mean_2019, mean_2020 )
    logger.info("T-test result for happiness scores 2019 vs 2020: t=%.2f, p=%.4f", t_stat1, p_val1)
    logger.info("Because the p-value is greater than 0.05, we fail to reject the null hypothesis. Therefore, we do not have sufficient evidence to conclude that there was a statistically significant difference in happiness scores between 2019 and 2020.")

    # second test
    latam = dataframe[dataframe['Regional indicator'] == 'Latin America and Caribbean']['Happiness score']
    na = dataframe[dataframe['Regional indicator'] == 'North America and ANZ']['Happiness score']
    t_stat, p_val = stats.ttest_ind(latam, na)
    logger.info("T-test result for happiness scores in Latin America/Caribean vs North America/ANZ: t=%.2f, p=%.4f", t_stat, p_val)
    logger.info('Since the p-value is less then 0.05, we reject the null hypothesis. This means that there is evidence to suggeest a statistically significan difference between happiness scores in North America vs Latin America')

@task
def correlation(dataframe):
    logger = get_run_logger()
    numeric_columns = dataframe.select_dtypes(include='number').columns

    explanatory_variables = [
        col for col in numeric_columns
        if col not in ['Happiness score', 'Ranking']
    ]

    alpha = 0.05
    number_of_tests = len(explanatory_variables)
    adjusted_alpha = alpha / number_of_tests

    logger.info(f'number of correlation tests: {number_of_tests}')
    logger.info(f'adjusted alpha: {adjusted_alpha}')
    correlation_results = {}

    for variable in explanatory_variables:

        clean_data = dataframe[
            [variable, 'Happiness score']
        ].dropna()

        correlation, p_value = pearsonr(
            clean_data[variable],
            clean_data['Happiness score']
        )

        correlation_results[variable] = {
            'correlation': correlation,
            'p_value': p_value
        }

        logger.info(
            f'{variable}: r={correlation:.4f}, p={p_value:.4f}, '
            f'significant at 0.05={p_value < alpha}, '
            f'significant after Bonferroni={p_value < adjusted_alpha}'
        )

    return correlation_results

@task
def summary(dataframe, corr_results):
    logger = get_run_logger()

    num_countries = dataframe['Country'].unique().size
    num_years = dataframe['year'].unique().size
    logger.info(f'Number of countres {num_countries} and there are {num_years} years')

    avg_regions_happy = dataframe.groupby('Regional indicator')['Happiness score'].mean()
    top_3 = avg_regions_happy.sort_values(ascending=False).head(3)
    bottom_3 = avg_regions_happy.sort_values().head(3)
    logger.info(f'Top 3 regions by happiness score: {top_3}')
    logger.info(f'Bottom 3 regions by happiness score: {bottom_3}')

    logger.info("We do not have sufficient evidence to conclude that there was a statistically significant difference in happiness scores between 2019 and 2020.")

    adjusted_alpha = 0.05 / len(corr_results)

    significant_correlations = {
        variable: result
        for variable, result in corr_results.items()
        if result['p_value'] < adjusted_alpha
    }

    if significant_correlations:
        strongest_variable = max(
            significant_correlations,
            key=lambda variable: abs(
                significant_correlations[variable]['correlation']
            )
        )

        strongest_correlation = significant_correlations[
            strongest_variable
        ]['correlation']

        logger.info(
            f'The variable most strongly correlated with happiness score '
            f'after Bonferroni correction was {strongest_variable} '
            f'(r={strongest_correlation:.4f})'
        )
    else:
        logger.info(
            'No variables had a statistically significant correlation '
            'with happiness score after Bonferroni correction.'
        )


@flow
def happiness_pipeline():
    df = load_data()
    descriptive_stats(df)
    visuals(df)
    hypo_test(df)
    correlation_results = correlation(df)
    summary(df, correlation_results)


if __name__ == "__main__":
    happiness_pipeline()
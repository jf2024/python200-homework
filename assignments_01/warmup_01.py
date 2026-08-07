import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# --- Part 1: Pandas ---

# Pandas Q1
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"First Three Rows are:\n {df.head(3)}")
print("------------------------")
print(f"The shape is: {df.shape}")
print("------------------------")
print(f"Data Types are: {df.dtypes}")
print("===========================================")

# Pandas Q2
q2_filter = df[(df['passed'] == True) & (df['grade'] > 80)]
print(q2_filter)
print("===========================================")

# Pandas Q3
df['grade_curved'] = df['grade'] + 5
print(df)
print("===========================================")

# Pandas Q4
df['name_upper'] = df['name'].str.upper()
print(df[['name', 'name_upper']])
print("===========================================")

# Pandas Q5
print(df.groupby('city')['grade'].mean())
print("===========================================")

#Pandas Q6
df["city"] = df["city"].replace("Austin", "Houston")
print(df)

#Pandas Q7
print(df.sort_values("grade", ascending=False))

# --- Part 1: Numpy ---

# Numpy Q1
arr1 = np.array([10, 20, 30, 40, 50])
print(f"The shape is: {arr1.shape}")
print("------------------------")
print(f"Data Type is: {arr1.dtype}")
print("------------------------")
print(f"Dimension is:\n {arr1.ndim}")
print("===========================================")


# Numpy Q2
arr2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(f"The shape is: {arr2.shape}")
print("------------------------")
print(f"Size is: {arr2.size}")
print("===========================================")


# Numpy Q3
print(arr2[0:2, 0:2])
print("===========================================")

# Numpy Q4
arr_zeros = np.zeros((3, 4))
print(arr_zeros)
print("------------------------")
arr_ones = np.ones((2, 5))
print(arr_ones)
print("===========================================")

# Numpy Q5
arr_step = np.arange(0, 50, 5)
print(arr_step)
print('Shape:', arr_step.shape)
print("Mean:", np.mean(arr_step)) 
print("Sum:", np.sum(arr_step)) 
print("Standard Deviation:", np.std(arr_step)) 
print("===========================================")

# Numpy Q6
arr_stan = np.random.normal(size=200)
print("Mean:", np.mean(arr_stan))
print("Standard Deviation:", np.std(arr_stan))

# --- Part 1: Matplotlib ---

# Matplot Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplot Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title("Scores by Subjects")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplot Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
plt.scatter(x1, y1, color='red')
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.xlabel("X Points")
plt.ylabel("Y Points")
plt.scatter(x2, y2, color='blue')
plt.legend(['First Points', 'Second Points'])
plt.show()

# Matplot Q4
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title("Basic X-Y Scatterplot")
plt.subplot(1, 2, 2)
plt.bar(subjects, scores)
plt.title("Scores by Subjects")
plt.tight_layout()
plt.show()

# --- Part 1: Descriptive Statistics ---

# DS Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print("Mean: ", np.mean(data))
print("Median: ", np.median(data))
print("Variance: ", np.var(data) )
print("Standard Deviation: ", np.std(data))
print("===========================================")

# DS Q2
arr_rand = np.random.normal(65, 10, 500)
plt.hist(arr_rand, bins=20, color="red", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Numbers")
plt.ylabel("Frequencies of Numbers")
plt.show()

# DS Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Value")
plt.show()

# DS Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparsion")
plt.ylabel("Value")
plt.show()

"""
From looking at the boxplots, exponential seems to be more skewed, top whisker much longer then the bottom whisker
and the median line is not centered. The descriptive statistic for the normal that we can use is mean since there doesn't 
seem to be any outliers while for exponential we will use the median due to its skewness
"""

# DS Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Mean Data 1: ", np.mean(data1) )
print("Median Data 1: ", np.median(data1) )
print("Mode Data 1: ", stats.mode(data1))

print("Mean Data 2: ", np.mean(data2))
print("Median Data 2: ", np.median(data2) )
print("Mode Data 2: ", stats.mode(data2) )

"""
The mean is very different to the outlier that is 150 for data2, skewing the data which causes
a big change in the mean. The median is the same for both though.
"""

# --- Part 1: Hypothesis Testing---

# # HT Q1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print("t-statistic:", t_stat)
print("p-value:", p_val)

# # HT Q2
if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")

print("===========================================")

# # HT Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat, p_val = stats.ttest_ind(before, after)
print("t-statistic:", t_stat)
print("p-value:", p_val)
print("===========================================")

# # HT Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores, 70)
print("t-statistic:", t_stat)
print("p-value:", p_val)
print("There doesn't appear to be significant difference between the scores and the benchmark of 70.")
print("===========================================")

# # HT Q5
t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative="less")
print("p-value:", p_val)
print("===========================================")

# # HT Q6
print(
"""
The difference between the groups is unlikely to be due to random chance because our p-value 
(0.0000015471) is less than 0.05. 
Therefore, we reject the null hypothesis that 
there is no difference between Group A and Group B. 
This suggests that there is a statistically significant difference between the two groups.
""")

# --- Part 1: Correlation---

# C Q1 
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
print(corr_matrix[0, 1])
print("===========================================")
"""
i expected the correlation to be 1 since the two groups seem to be moving up 
(the higher the x, y also increases), but doesnt show causation
"""

# C Q2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
r, p = pearsonr(x, y)
print("Correlation:", round(r, 2))
print("p-value:", round(p, 4))
print("===========================================")

# CQ3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(df.corr())
print("===========================================")

# CQ4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y, color="green")
plt.title("Negative Correlation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# CQ5
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- Part 1: Pipelines---

# Pipelines Q1

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr)

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    series_info = {
        "mean": np.mean(series),
        "median": np.median(series),
        "std": np.std(series),
        "mode": series.mode()[0]
    }

    return series_info

def data_pipeline(arr):
    values = create_series(arr)
    cleaned = clean_data(values)
    summary = summarize_data(cleaned)
    return summary

print(data_pipeline(arr))

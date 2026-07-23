import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Part 1: Pandas ---

# Pandas Q1
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# print(f"First Three Rows are:\n {df.head(3)}")
# print("------------------------")
# print(f"The shape is: {df.shape}")
# print("------------------------")
# print(f"Data Types are: {df.dtypes}")
# print("===========================================")

# Pandas Q2
# q2_filter = df[(df['passed'] == True) & (df['grade'] > 80)]
# print(q2_filter)
# print("===========================================")

# Pandas Q3
# df['grade_curved'] = df['grade'] + 5
# print(df)
# print("===========================================")

# Pandas 04
# df['name_upper'] = df['name'].str.upper()
# print(df[['name', 'name_upper']])
# print("===========================================")

# Pandas 05
# print(df.groupby('city')['grade'].mean())
# print("===========================================")

#Pandas 06
# df["city"] = df["city"].replace("Austin", "Houston")
# print(df)

#Pandas 07
#print(df.sort_values("grade", ascending=False))

# --- Part 1: Numpy ---

# Numpy Q1
# arr1 = np.array([10, 20, 30, 40, 50])
# print(f"The shape is: {arr1.shape}")
# print("------------------------")
# print(f"Data Type is: {arr1.dtype}")
# print("------------------------")
# print(f"Dimension is:\n {arr1.ndim}")
# print("===========================================")


# Numpy Q2
arr2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# print(f"The shape is: {arr2.shape}")
# print("------------------------")
# print(f"Size is: {arr2.size}")
# print("===========================================")


# Numpy Q3
# print(arr2[0:2, 0:2])
# print("===========================================")

# Numpy Q4
# arr_zeros = np.zeros((3, 4))
# print(arr_zeros)
# print("------------------------")
# arr_ones = np.ones((2, 5))
# print(arr_ones)
# print("===========================================")

# Numpy Q5
# arr_step = np.arange(0, 50, 5)
# print(arr_step)
# print('Shape:', arr_step.shape)
# print("Mean:", np.mean(arr_step)) 
# print("Sum:", np.sum(arr_step)) 
# print("Standard Deviation:", np.std(arr_step)) 
# print("===========================================")

# Numpy Q6
# arr_stan = np.random.normal(size=200)
# print("Mean:", np.mean(arr_stan))
# print("Standard Deviation:", np.std(arr_stan))

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
# plt.bar(subjects, scores)
# plt.title("Scores by Subjects")
# plt.xlabel("Subjects")
# plt.ylabel("Scores")
# plt.show()

# Matplot Q3
# x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
# plt.scatter(x1, y1, color='red')
# x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
# plt.xlabel("X Points")
# plt.ylabel("Y Points")
# plt.scatter(x2, y2, color='blue')
# plt.legend(['First Points', 'Second Points'])
# plt.show()

# Matplot Q4
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title("Basic X-Y Scatterplot")
plt.subplot(1, 2, 2)
plt.bar(subjects, scores)
plt.title("Scores by Subjects")
plt.tight_layout()
plt.show()

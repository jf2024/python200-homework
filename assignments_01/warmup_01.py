import pandas as pd

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
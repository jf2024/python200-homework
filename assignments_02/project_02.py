import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns

# Paramter i would need to add is "sep"
    # similar to what we did in project1 

# Task 1
data = pd.read_csv('../data/student_performance_math.csv', sep=';')
print('Shape is:', data.shape)
print(data.head(5))
print(data.info())

plt.hist(data['G3'], bins=21, color="purple", edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Math Final Grade Score (0-20)")
plt.ylabel("Number of Students")
plt.savefig('outputs/g3_distribution')
plt.show()

# Task 2
print('Before cleaning, shape is', data.shape)
clean_data = data[data['G3'] != 0].copy()
print('After cleaning, shape is', clean_data.shape) #38 rows/students removed due to G3 score being 0
# we remove these because they dont tell us anything in our data, can show bias in our model that students who
    # did take the examples got a 0 when in reality they didnt take it and got a 0 by default in the data

yes_no_cols = ['schoolsup', 'internet', 'higher', 'activities']

for col in yes_no_cols: #https://stackoverflow.com/questions/40901770/is-there-a-simple-way-to-change-a-column-of-yes-no-to-1-0-in-a-pandas-dataframe
    clean_data[col] = clean_data[col].map({'no': 0, 'yes': 1})

clean_data['sex'] = clean_data['sex'].map({'F': 0, 'M': 1})

print('Correlation of absences and original data', pearsonr(data['absences'], data['G3'])[0] ) #0 is correlation, 1 is the p-valie
print('Correlation of absences and cleaned data', pearsonr(clean_data['absences'], clean_data['G3'])[0] )
# In the original data, students didnt take the exam so having a 0 was misspresenting them, without them we get a clearer
    # picture that perphaps absences will make a student perform worse which makes sense

# Task 3
numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc',
                    'schoolsup', 'internet', 'higher', 'activities']
corr_list = []
for f in numeric_features:
    corr = round(pearsonr(clean_data[f], clean_data['G3'])[0], 2)
    corr_list.append((f, corr))

print(sorted(corr_list, key=lambda tup: tup[1])) #https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index
# Strongest negative relationship is failures then schoolsup, then absences, then walc, goout, age --> all correlate in the negative direction 
# Strongest positive relationship is Medu, then Fedu, and Studytime 
# A little shocked that parents education matter more then studytime but its not that big of a margin and shocked that age has a negative affect

sns.boxplot(x='age', y='G3', data=clean_data)

plt.title("Final Math Score by Age")
plt.xlabel("Age")
plt.ylabel("Final Math Score (0-20)")
plt.savefig("outputs/age_vs_g3_boxplot.png")
plt.show()
# Looks like the younger students (15-16), perform better or the same for the older kids
    # with the exception of age 20 while age 21 and 22 dont have enough students to rlly show a boxplot

sns.boxplot(x='studytime', y='G3', data=clean_data)
plt.title("Final Math Score by Study Time")
plt.xlabel("Study Time")
plt.ylabel("Final Math Score (0-20)")
plt.savefig("outputs/studytime_vs_g3_boxplot.png")
plt.show()
# Study time looks pretty even across the board between the final score being at 11-13 on average
    # though very small increase in score if studying for at least 30 min to 1 hour

# Task 4
X = clean_data[['failures']] # dont forget to rehspae if just 1 feature
y = clean_data['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train) #here we train the model with the training data
y_pred = model.predict(X_test)  #testing the model on the test set

print("Slope:", model.coef_[0])

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)
# For every failure in the class, the student's final score will decrease about 1.42 points
# RMSE of 2.96 means our predictions are off by about 3 points 
# Honestly pretty low but considering failure would have a negative affect, the fact that the R2 didn't go below
    # 0 for the test set is a good sign

print("=" * 60)

# Task 5
feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]

X = clean_data[feature_cols].values
y = clean_data["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train) 

y_pred = model.predict(X_test)  
y_pred_train = model.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_train = r2_score(y_train, y_pred_train)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² test:", r2)
print("R² train:", r2_train)
# Adding more features slightly helped, our R2 improved from 0.09 to 0.26, so an increase 
# Our RMSE decreased from 2.96 to 2.66 but the jump in improvement isn't that large 


for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
#activities (-0.059), absences(0.061), traveltime(-0.083) all near 0, so can drop these
#want to keep studytime (0.311), failures(-.800), goout(-0.313), schoolsup(-2.263), medu(0.163), fedu(0.187)

# Task 6

plt.scatter(y_pred, y_test)

min_score = min(y_pred.min(), y_test.min())
max_score = max(y_pred.max(), y_test.max())

plt.plot([min_score, max_score], [min_score, max_score], color="red", linestyle="--")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3 Score")
plt.ylabel("Actual G3 Score")

plt.savefig("outputs/predicted_vs_actual_g3.png")
plt.show()

# Size of filtered dataset was 357 students while original data was 395 students
# Full Feature Model, RMSE of 2.66 so on average, predictions off by 2.66 points while the R2 was 0.26
# Largest impact that I saw was schoolsup with -2.263 which means if additional support and holding the other
    # variables constant, score would decrease 2.2 points compared to a student with no additional sport (since they dont need it)
# In addition internet got a 1.037 so holding the other variable constants, with internet access seems to be an association
    # with gaining at least 1 point for the final math score  
# Surprised that the school support was the most negative and that internet was the most positive for my model

print("=" * 60)

# Neglected Feature
feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex", 'G1']

X = clean_data[feature_cols].values
y = clean_data["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE", rmse)
print("New R2 with G1 added as a feature", r2)
# A high R2 doesn't mean G1 causes G3, can say there is some relationship/association with it.
    # A good feature for the model but doesn't help in practical terms since we don't know G1 
    # when the school year begins. Can maybe use it after the G1 scores to see which students are struggling though
    # and give more attention. 
# Look at other behaviors like studytime, safe home environment, etc... 
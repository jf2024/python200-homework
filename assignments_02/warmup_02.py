import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split

# --- scikit-learn API ---

## Q1
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()                # 1. create model
model.fit(years, salary)                 # 2. fit model to data (learn)

new_years = np.array([4, 8]).reshape(-1, 1)     # our new data
y_predicted = model.predict(new_years)            # 3. predict with new data
print(f"The slope is {model.coef_[0]}")
print(f"The intercept is {model.intercept_}")
print(f"Someone with 4 years of experience will earn about ${y_predicted[0].round(0)}")
print(f"Someone with 8 years of experienee will earn about ${y_predicted[1].round(0)}")
print("=" * 60)

## Q2
x = np.array([10, 20, 30, 40, 50])
print("Current x shape is", x.shape)
new_X = x.reshape(-1, 1)
print("New x shape is now", new_X.shape)
# The reason we need to reshape it to a 2D instead of keeping it as a 
    # 1d array is because for scikit-learn, the input x has to be a 2d shape which 
    # means (number of samples, number of features). Can think of it like a matrix
    # where each row is one sample/data point and the columns are the features
        # So in the above example, when we reshape it, we have 5 rows (observations) and
            # just one column (feature)
print("=" * 60)

##  Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)  
kmeans.fit(X_clusters)                                  
labels = kmeans.predict(X_clusters)   

centroids = kmeans.cluster_centers_
print("Cluster centers", kmeans.cluster_centers_)

print("Number of points in each cluster", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)

plt.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='X', color='black', s=100, label='Centroids'
)

plt.title("Optimal Number of Clusters")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.savefig('outputs/kmeans_clusters.png')
plt.show()

# --- Linear Regression ---
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

## Q1
plt.scatter(age, cost, c=smoker, cmap='coolwarm')
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost ($)")
plt.savefig('outputs/cost_vs_age.png')
plt.show()
# Looking at the plot, people that smoke tend to pay more in medical
    # in costs then non-smokers across the ages, the smoker variable plays
    # a crucial role 
print("=" * 60)

## Q2
X = age.reshape(-1, 1) #our one feature
y = cost #our target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Shapes for x and y: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}")
print("=" * 60)

## Q3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)
print("RMSE", rmse)
print('R^2', r2)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
# As age increases, the medical cost annually will increase by about $196
print("=" * 60)

## Q4
X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)
print("R²:", model_full.score(X_test_f, y_test_f))
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
# The R2 increases quite a bit when adding the smoker feature (now at 0.77), doing much better then what it was doing
    # previously (at a measly 0.069)
# If a person is a smoker, it will increase the medical annual cost by about $14538 while holding the age fixed. 
print("=" * 60)

## Q5
y_pred = model_full.predict(X_test_f)

plt.scatter(y_test_f, y_pred)

plt.plot(
    [y_test_f.min(), y_test_f.max()],
    [y_test_f.min(), y_test_f.max()]
)
plt.xlabel("Actual ($)")
plt.ylabel("Predicted ($)")
plt.title("Predicted vs Actual Medical Annual Costs")
plt.savefig('outputs/predicted_vs_actual_cost.png')
plt.show()
# A point above the diagonal means we overpredicted the data point (in this case, predicting a person
    # will pay more but in reality will pay less).
# A point below te diagonal is the opposite, we underpredicted the data point (predicting a person
    # won't pay too much but in reality will pay a bit more). 
# And on the diagonal is the perfect prediction 



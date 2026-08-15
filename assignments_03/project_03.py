import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Task 1
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())
print(df.info())
# We have 4601 emails, 57 features + the 1 target variable which is spam_label
print("Frequencies of our target variable", df['spam_label'].value_counts(normalize=True))
# about 60% non-spam and 39% spam --> a little bit imbalacned but not too severely 
# for intepretation of raw accuracy score, we can't rely on it because a model 
    #  can have high accuracy just by predicting the most common class. 
    # Precision/recall help us see whether the model is actually learning about the different classes. 

features = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']

for feature in features: #getting help from chatgpt for this
    plt.figure()
    plt.boxplot([
        df[df['spam_label'] == 0][feature],
        df[df['spam_label'] == 1][feature]
    ], label=['Ham', 'Spam'])

    title = f"{feature}: Ham vs Spam"
    plt.title(title)
    plt.ylabel(feature)
    plt.savefig(f"outputs/{feature}_ham_vs_spam.png")
    plt.show()
# Looks like Spam just slightly edges out non-spam in all three graphs on average but most of the distributions are at 
    # 0 with lots of outliers, not really any differences

print(df.describe())
# lots of email are in the 0's, while other features are rlly big (capital_run_length_total).
# for models such as KNN and/or logistic regression (distance-based) it matters a lot so 
    # we need to scale but for decision trees it does not

# Task 2
X = df.drop(columns=['spam_label'])
y = df['spam_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_var)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance vs. Number of Components")
plt.savefig("outputs/pca_variance_explained_project03.png")
plt.show()


n = np.argmax(cumulative_variance >= 0.90) + 1
print(f"number of components for 90% variance: {n}")

X_train_pca = X_train_pca[:, :n] #reduced pca arrays
X_test_pca = X_test_pca[:, :n] #reduced pca arrays

# Task 3
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)
print("Accuracy for knn unscaled:", accuracy_score(y_test, y_pred_knn_unscaled))
print(classification_report(y_test, y_pred_knn_unscaled)) #0.79 acc

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy for knn scaled:", accuracy_score(y_test, y_pred_knn_scaled)) #0.906
print(classification_report(y_test, y_pred_knn_scaled))

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print("KNN - PCA")
print("Accuracy for knn with pca:", accuracy_score(y_test, y_pred_knn_pca)) # 0.907
print(classification_report(y_test, y_pred_knn_pca))


depths = [3, 5, 10, None]
for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Decision Tree - max_depth={depth}")
    print("Training accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)
    print()


# None is the highest at 99 on training and 91 on testing
# as depth increases we see both the train and test data increase but more of an 
    # increase for training, this is an overfitting problem
    # so we should stick with 10 as it has the highest of both but not a perfect by any means

chosen_depth = 10
tree = DecisionTreeClassifier(
    max_depth=chosen_depth,
    random_state=42
)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("Final Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_tree)) #0.908
print(classification_report(y_test, y_pred_tree))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy for random forest:", accuracy_score(y_test, y_pred_rf)) #0.945
print(classification_report(y_test, y_pred_rf))

lr_scaled = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='liblinear'
)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_lr_scaled = lr_scaled.predict(X_test_scaled)
print("Accuracy for logistic regressoin scaled:", accuracy_score(y_test, y_pred_lr_scaled)) #0.929
print(classification_report(y_test, y_pred_lr_scaled))

lr_pca = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='liblinear'
)
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
print("Accuracy for logistic regressoin pca:", accuracy_score(y_test, y_pred_lr_pca)) #0.918
print(classification_report(y_test, y_pred_lr_pca))

# Random forest performs the best, probably due to its feature based approach instead of distanced based
# PCA doesn't seem to beat the full feature models which makes sense since trees dont benefit from it while 
    # logistic regression and knn improve slightly but not considerably 
# No, accuracy wouldnt be the right metric. I think false positives are worse since if its a real email and its marked as spam,
    # the person wont ever read it. With that in mind, we should favor precision (of everything that was called spam, how much was actually spam)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    display_labels=["Ham", "Spam"]
)
# The type of error that the model makes most often would be false negatives, meaning in 32 attempts/times
    # it marked spam messages as non-spam

plt.title("Random Forest Confusion Matrix")
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.show()

tree_importances = pd.Series(
    tree.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print("decision tree features:")
print(tree_importances.head(10))

rf_importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print("random forest features:")
print(rf_importances.head(10))

# Model seesms to agree on which features are the most important (the top 3 is the same in differnt order)
# And yes intuition matches what makes an email spam

top10 = rf_importances.head(10).sort_values()
plt.figure(figsize=(8, 6))
top10.plot(kind='barh')
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.show()

# Task 4
models = {
    "KNN Unscaled": KNeighborsClassifier(n_neighbors=5),
    "KNN Scaled": KNeighborsClassifier(n_neighbors=5),
    "KNN PCA": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Logistic Regression Scaled": LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='liblinear'
    ),
    "Logistic Regression PCA": LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='liblinear'
    )
}

data = {
    "KNN Unscaled": (X_train, y_train),
    "KNN Scaled": (X_train_scaled, y_train),
    "KNN PCA": (X_train_pca, y_train),
    "Decision Tree": (X_train, y_train),
    "Random Forest": (X_train, y_train),
    "Logistic Regression Scaled": (X_train_scaled, y_train),
    "Logistic Regression PCA": (X_train_pca, y_train)
}

for name, model in models.items():
    
    X_data, y_data = data[name]
    
    scores = cross_val_score(
        model,
        X_data,
        y_data,
        cv=5,
        scoring='accuracy'
    )
    
    print(name)
    print("Fold scores:", scores)
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print()
# Random Forest was the most accurate model with a mean accuracy of 95.43%. 
# Logistic Regression with PCA was the most stable because it had the lowest standard deviation across the five folds. 
# Overall, the cross-validation results were similar to the single train/test split, 
    # with Random Forest remaining the best-performing model.
# Results pretty similar to the original ones from task3

# Task 5
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
print("Random Forest Pipeline")
print(classification_report(y_test, rf_pred))

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ))
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
print("Logistic Regression Pipeline")
print(classification_report(y_test, lr_pred))

# The pipelines don't have the same structure, with random forest
    # scaling is not needed while we need it for logistic regression since it helps
# The benefit is that we dont have to separate the preprocessing step, we can include it in the pipeline
    # and it does it for us and we don't have to worry about X_train_scaled, X_test_scaled and it helps 
    # prevent data leakage sicne less room for error
    # also makes it easier for others to read the code and run it without needed to remember all the preprocessing
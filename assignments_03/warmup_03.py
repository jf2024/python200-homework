import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# ---- Preprocessing ----

## Q1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Shapes for x and y: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}")

## Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learns mean and std from training data only
X_test_scaled  = scaler.transform(X_test)        # applies the same scaling to test data

print('Mean of each column: ',X_train_scaled.mean(axis=0)) #https://stackoverflow.com/questions/31037298/pandas-get-column-average-mean
# We only fit the scaler on X_train only because we want our scaler to learn the underlying mean and std for our training data
    # if we were to do the same on the X_test, we would be using data from the test data which isn't good because this data
    # should be unseen (data leakage.) Instead, we just use transform on the X_test, which applies the same MEAN
    # and std that we learned from X_train, and now X_test is on the same scale as X_train without learning from the test data.
print("=" * 60)

# ---- KNN ----

## Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print("=" * 60)

## Q2
knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, y_train)

preds = knn2.predict(X_test_scaled)

print("Accuracy for scaled:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
# My accruracy actually decreased (went from 1 with the unscaled to 0.933), since the data seems to already
    # be in the same scale or at least similar scales, doing a standarirzation was unnecessary
print("=" * 60)

## Q3
knn3 = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn3, X_train, y_train, cv=5)

print(cv_scores)           # accuracy on each fold
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")
# I would say this result is more trustworthy b/c with each fold, we get different variations 
    # of the training data and we get an average over multiple folds/splits compared to just one
print("=" * 60)

## Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15] 

for k in k_values:
    knn4 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn4, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")
# I think I would choose 5 or 7 since they gave the highest mean though all the folds give a 
    # pretty high score with the lowest being fold 1 at 0.942
print("=" * 60)

# ---- Classifier Evaluation ----

##  Q1
cm = confusion_matrix(y_test, preds)
cm_dispaly = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names )
cm_dispaly.plot(colorbar=False)
plt.title("KNN Iris Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()
# The model gets a bit confused with versicolor and virginica but not too big of a deal

# ---- Decision Trees ----

## Q1
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print("Accuracy for decision tree:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, preds))
# Decision Tree accuracy is at 0.96 compared to the KNN (unscaled) at 1 and KNN (scaled) at 0.93. Pretty much the same
    # across the board and can't really go wrong with any of them for the final model.
# Since decision tree don't rely on distance calculations, I don't think the different scales of the features
    # would matter too much, decision trees split by one feature 
print("=" * 60)

# ---- Logistic Regression and Regularization ----

## Q1 - followed Prof. Tom Arns suggestion on the slack
for C in [0.01, 1.0, 100]:
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
        )
    )
    log_reg.fit(X_train_scaled, y_train)
    coef_sum = np.abs(np.vstack([est.coef_ for est in log_reg.estimators_])).sum()
    print(f"C={C}, total coefficient magnitude={coef_sum}")
# Regularization refers if there should be a penalty for high coefficeints in our model, we cant just take the training data
# at face value and soley trust it, can lead to overfitting
    # With a smaller C, we say that the training data doesn't represent the real world, we should penalize coefficients 
        # that are too big
    # A larger C says we trust the data and it dooesn't matter if the coefficients are large, as long as our prediction
        # is pretty accurate 
print("=" * 60)

# ---- PCA ----
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

## Q1
print('Shape of X_digits', X_digits.shape)
print('Shape of images', images.shape)
fig, axes = plt.subplots(1, 10, figsize=(15, 2))

for digit in range(10):
    # Find the first example of this digit
    index = list(y_digits).index(digit)

    axes[digit].imshow(images[index], cmap='gray_r')
    axes[digit].set_title(str(digit))
    axes[digit].axis('off')

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.show()

## Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.title('Digits Data PCA Components')
plt.colorbar(scatter, label='Digit')
plt.savefig("outputs/pca_2d_projection.png")
plt.show()
# Yes, same digit points do overlap, specifically 0, 6, and 4 are the easy ones to spot while the rest
    # seem to overlap quite a bit with just 

## Q3
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_var)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance vs. Number of Components")
plt.savefig("outputs/pca_variance_explained.png")
plt.show()
# To explain 80% variance, at least 12 components to about 15

## Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_comps = [2, 5, 15, 40]
fig, axes = plt.subplots(5, 5, figsize=(8, 8))

for i in range(5):
    axes[0, i].imshow(images[i], cmap="gray_r")
    axes[0, i].axis("off")

for row in range(4):
    n = n_comps[row]

    for i in range(5):
        reconstruction = reconstruct_digit(i, scores, pca, n)
        axes[row + 1, i].imshow(reconstruction, cmap="gray_r")
        axes[row + 1, i].axis("off")

plt.savefig("outputs/pca_reconstructions.png")
plt.show()
# digits become digits or look like numbers at around 14 or 15 components while its blurry for smaller number
    # of components. (which follows what the cumulative variance plot showed us earlier)
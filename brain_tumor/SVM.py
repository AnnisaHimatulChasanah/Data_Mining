import os
import cv2
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
import joblib

# Prepare/collect data
path = os.listdir(r'C:\Users\annis\Documents\brain_tumor\Training')
classes = {'no_tumor': 0, 'pituitary_tumor': 1}

X = []
Y = []

for cls in classes:
    pth = r'C:\Users\annis\Documents\brain_tumor\Training/' + cls
    for j in os.listdir(pth):
        img = cv2.imread(pth + '/' + j, 0)
        img = cv2.resize(img, (200, 200))
        X.append(img)
        Y.append(classes[cls])

# Reshape data
X = np.array(X)
Y = np.array(Y)
X_updated = X.reshape(len(X), -1)
# Visualize data
plt.imshow(X[0], cmap='gray')
# Split Data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=0.20)

# Feature Scaling
xtrain = xtrain / 255
xtest = xtest / 255

# Feature Selection: PCA
pca = PCA(0.98)
xtrain_pca = pca.fit_transform(xtrain)  # Apply PCA transformation during training
xtest_pca = pca.transform(xtest)        # Apply the same transformation during testing
joblib.dump(pca, 'pca_model.joblib')

# Train Model
sv = SVC()
sv.fit(xtrain_pca, ytrain)  # Train the model using the PCA-transformed training data
# Save the trained model using joblib
joblib.dump(sv, 'svm_model.joblib')

# Evaluation
print("Training Score (SVM):", sv.score(xtrain_pca, ytrain))
print("Testing Score (SVM):", sv.score(xtest_pca, ytest))

# Prediction for SVM
pred = sv.predict(xtest_pca)
misclassified = np.where(ytest != pred)
print("Total Misclassified Samples(SVM): ", len(misclassified[0]))

# TEST MODEL
dec = {0: 'No Tumor', 1: 'Positive Tumor'}

# Visualize SVM predictions on 'no_tumor' images
plt.figure(figsize=(12, 12))
plt.suptitle('Support Vector Machine Predictions - No Tumor')

# Testing SVM on 'no_tumor'
for c, i in enumerate(os.listdir(r'C:\Users\annis\Documents\brain_tumor\Testing\no_tumor')[:20], 1):
    plt.subplot(4, 5, c)  # Use 4 rows and 5 columns for a 20-image display
    img = cv2.imread(r'C:\Users\annis\Documents\brain_tumor\Testing\no_tumor' + i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1) / 255
    img1_pca = pca.transform(img1)  # Apply PCA transformation to the testing image
    prediction = sv.predict(img1_pca)
    plt.imshow(img, cmap='gray')
    plt.title(f"{dec[prediction[0]]}")
    plt.axis('off')


# Visualize SVM predictions on 'pituitary_tumor' images
plt.figure(figsize=(12, 12))
plt.suptitle('Support Vector Machine Predictions - Pituitary Tumor')

# Testing SVM on 'pituitary_tumor'
for c, i in enumerate(os.listdir(r'C:\Users\annis\Documents\brain_tumor\Testing\pituitary_tumor')[:20], 1):
    plt.subplot(4, 5, c)  # Use 4 rows and 5 columns for a 20-image display
    img = cv2.imread(r'C:\Users\annis\Documents\brain_tumor\Testing\pituitary_tumor' + i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1) / 255
    img1_pca = pca.transform(img1)  # Apply PCA transformation to the testing image
    prediction = sv.predict(img1_pca)
    plt.imshow(img, cmap='gray')
    plt.title(f"{dec[prediction[0]]}")
    plt.axis('off')


plt.show()

pred_prob = sv.decision_function(xtest_pca)  # Decision function scores for binary classification
fpr, tpr, thresholds = roc_curve(ytest, pred_prob)
roc_auc = roc_auc_score(ytest, pred_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Support Vector Machine')
plt.legend(loc='lower right')
plt.show()

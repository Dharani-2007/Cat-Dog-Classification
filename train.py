import cv2
import os
import numpy as np
import pickle

X = []
y = []

# Cats
cat_folder = "training_set/training_set/cats"

for image_name in os.listdir(cat_folder)[:2000]:
    image_path = os.path.join(cat_folder, image_name)

    img = cv2.imread(image_path)

    if img is not None:
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(0)

# Dogs
dog_folder = "training_set/training_set/dogs"

for image_name in os.listdir(dog_folder)[:2000]:
    image_path = os.path.join(dog_folder, image_name)

    img = cv2.imread(image_path)

    if img is not None:
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(1)
print("Number of images:", len(X))
print("First image shape:", X[0].shape)
X = np.array(X)
y = np.array(y)

print("Total Images:", len(X))
print("Cats:", np.sum(y == 0))
print("Dogs:", np.sum(y == 1))

# Flatten images
X = X.reshape(X.shape[0], -1)

print("Flattened Shape:", X.shape)

# Scale data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))

# Train SVM
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)
print("\nTraining Model...")
model.fit(X_train, y_train)
print("Model Training Completed!")

# Test
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save model
pickle.dump(model, open("cat_dog_model.pkl", "wb"))

# Save scaler too
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel Saved Successfully!")
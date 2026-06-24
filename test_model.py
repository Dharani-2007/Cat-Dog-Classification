import cv2
import pickle
import os

# Load model
model = pickle.load(open("cat_dog_model.pkl", "rb"))

correct = 0
total = 0

# Test Cats
cat_folder = "test_set/test_set/cats"

for image_name in os.listdir(cat_folder)[:100]:

    image_path = os.path.join(cat_folder, image_name)

    img = cv2.imread(image_path)

    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = img.reshape(1, -1)

        prediction = model.predict(img)

        if prediction[0] == 0:
            correct += 1

        total += 1

# Test Dogs
dog_folder = "test_set/test_set/dogs"

for image_name in os.listdir(dog_folder)[:100]:

    image_path = os.path.join(dog_folder, image_name)

    img = cv2.imread(image_path)

    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = img.reshape(1, -1)

        prediction = model.predict(img)

        if prediction[0] == 1:
            correct += 1

        total += 1

accuracy = (correct / total) * 100

print("Total Tested Images:", total)
print("Correct Predictions:", correct)
print("Test Accuracy:", accuracy, "%")
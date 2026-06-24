import cv2
import pickle
import numpy as np

# Load model
model = pickle.load(open("cat_dog_model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Select image
# Select image
image_path = "test_set/test_set/cats/cat.4002.jpg"

# Read image
img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

print("Image Shape:", img.shape)

# Resize
img = cv2.resize(img, (64, 64))

# Flatten
img = img.reshape(1, -1)

# Scale
img = scaler.transform(img)

# Predict
prediction = model.predict(img)

print("Raw Prediction:", prediction)

if prediction[0] == 0:
    result = "Dog"
else:
    result = "Cat"

print("Prediction:", result)

# Show result
# Read image again for display
img_display = cv2.imread(image_path)

# Resize display image
img_display = cv2.resize(img_display, (500, 500))

cv2.putText(
    img_display,
    f"Prediction: {result}",
    (10, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 255, 0),
    2
)

cv2.imshow("Image Classification Result", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
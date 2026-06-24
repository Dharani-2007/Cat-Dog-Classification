Project Overview

Cat-Dog Classification is a Machine Learning and Computer Vision project that automatically identifies whether an image contains a cat or a dog. The system uses image preprocessing techniques and a Support Vector Machine (SVM) classifier trained on labeled cat and dog images.
This project demonstrates the application of Machine Learning in image recognition and classification tasks.

Problem Statement

Manually identifying and categorizing large collections of animal images is time-consuming and inefficient.
The objective of this project is to develop an automated image classification system capable of accurately predicting whether an image belongs to the cat or dog category.

Features

Image Classification using Machine Learning

Automatic Cat and Dog Detection

OpenCV-based Image Processing

SVM Classification Model

Image Resizing and Feature Extraction

Real-time Prediction from Input Images

Simple and Easy-to-Use Interface

Dataset

The project uses the Cats and Dogs image dataset.

Dataset Structure:

training_set/

├── cats/

├── dogs/

test_set/

├── cats/

├── dogs/

Training Images:

Cats: 2000

Dogs: 2000

Total Images:
4000

Technologies Used

Python 

OpenCV

NumPy

Scikit-Learn

Pickle

Methodology

1. Data Collection
Cat and Dog images are collected from the dataset and organized into separate folders.
2. Image Preprocessing
Images are loaded using OpenCV
Resized to 64 × 64 pixels
Converted into numerical arrays
3. Feature Extraction
Images are flattened into one-dimensional feature vectors for training.
4. Data Normalization
StandardScaler is used to normalize image features.
5. Model Training
An SVM classifier with RBF kernel is trained using the processed dataset.
6. Prediction
New images are processed and passed to the trained model to predict:
Cat
Dog

Model Performance

The trained SVM model achieved:

Accuracy: 65.25%
Training Images: 3200
Testing Images: 800
Confusion Matrix:
[[276 124]
 [154 246]]

 Project Structure

 Cat-Dog-Classification/
│
├── train.py
├── predict.py
├── test_model.py
├── scaler.pkl
├── .gitignore
└── README.md

Installation
Install required libraries:
pip install opencv-python numpy scikit-learn

Usage
Train the Model
python train.py
This trains the SVM model and generates:
cat_dog_model.pkl
scaler.pkl
Predict an Image
python predict.py
Sample Output:
Prediction: Cat
or
Prediction: Dog

Sample Results
Dog Image
Input: dog.4002.jpg
Prediction: Dog
Cat Image
Input: cat.4001.jpg
Prediction: Cat

Applications
Animal Image Classification
Wildlife Monitoring
Veterinary Assistance Systems
Smart Surveillance Systems
Educational AI Projects

Future Improvements
Deep Learning using CNN
TensorFlow/Keras Integration
Higher Accuracy Models
Web-based Interface using Streamlit
Multi-Animal Classification
Real-time Webcam Detection


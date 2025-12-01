# Handwritten Digit Recognition Using AI (MNIST Dataset)

A Deep Learning Project for CAP 4630 – Intro to Artificial Intelligence

# Overview

This project builds an end-to-end AI system that recognizes handwritten digits (0–9) using the MNIST dataset.
Two models were implemented and compared:

Baseline Fully Connected Neural Network (Dense NN)

Convolutional Neural Network (CNN)

The goal is to demonstrate how model architecture impacts performance on image classification tasks and to visualize common errors using a confusion matrix and misclassified digits.

# Objectives

Understand and explore the MNIST image dataset

Preprocess grayscale pixel data

Train and evaluate a baseline Dense Neural Network

Train and evaluate a Convolutional Neural Network

Compare performance using accuracy and error analysis

Visualize results using plots, confusion matrix, and misclassified examples

Summarize findings and discuss limitations & future improvements

# Dataset: MNIST
Property	Value
Total Images	70,000
Training Set	60,000
Test Set	10,000
Image Size	28 × 28 pixels
Channels	Grayscale (1)
Classes	10 (digits 0–9)

The MNIST dataset is widely used as a benchmark for evaluating image recognition models.

# Technologies & Libraries

Python 3.9+

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook / VS Code

# Model Architectures
1. Baseline Model – Fully Connected NN

Flattened input (784 units)

Dense(128, ReLU)

Dense(10, Softmax)

Accuracy: ~97%

2. Convolutional Neural Network (CNN)

Conv2D(32 filters) → MaxPooling2D

Conv2D(64 filters) → MaxPooling2D

Flatten

Dense(64, ReLU)

Dense(10, Softmax)

Accuracy: ~98–99%

The CNN outperformed the baseline significantly due to its ability to learn spatial features such as edges, curves, and stroke patterns.

# Results Summary
Model	Test Accuracy
Baseline Dense NN	~0.97
CNN	~0.99
# Error Analysis

Confusion matrix reveals small confusion between similar digits.

Misclassified examples often include ambiguous or messy handwriting.

(Insert your plots here if uploading to GitHub)

-Baseline accuracy graph

-CNN accuracy graph

-Confusion matrix

-Misclassified digit samples

# How to Run This Project
1. Clone the Repository
git clone https://github.com/RedrovanD/mnist-handwritten-digit-ai.git
cd mnist-handwritten-digit-ai

2. Create a Virtual Environment (Recommended)
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows

3. Install Requirements
pip install -r requirements.txt

 or

install manually: pip install tensorflow numpy matplotlib seaborn scikit-learn

4. Run the Notebook
jupyter notebook mnist_handwritten_digit_classifier.ipynb

 or

open it in VS Code with the Jupyter extension.

# Project Structure
│── mnist_project.ipynb          # Full implementation

│── README.md                    # Project documentation

│── Requirements.txt             # Dependencies

# Project Demo / Presentation

https://your-link-here.com

# Limitations

Trained only on MNIST (clean, centered digits)

Does not generalize to real-world handwriting without augmentation

Only recognizes digits, not letters or sequences of text

# Future Enhancements

Apply data augmentation (rotation, shifting, noise)

Experiment with deeper CNN architectures

Try EMNIST to classify letters as well

Extend to full handwritten words

Use transfer learning for more complex datasets

# Author

David Redrovan

CAP 4630 – Intro to Artificial Intelligence

Fall 2025

# Handwritten Digit Recognition Using AI (MNIST Dataset)

This project builds and evaluates an AI system capable of classifying handwritten digits (0–9) using the MNIST dataset, one of the most widely used benchmarks in machine learning and deep learning.

This work was completed as the Final Project for CAP 4630 – Intro to Artificial Intelligence.

# Project Overview

Handwritten digit recognition is a classic AI problem used in postal systems, bank check processing, and form digitization. The goal of this project is to design an AI model that can read 28×28 grayscale images of handwritten digits and correctly identify the corresponding digit.

This project compares two different machine learning approaches:

Baseline Fully Connected Neural Network

Convolutional Neural Network (CNN)

# Objectives

Load and explore the MNIST dataset

Preprocess grayscale image data

Train a baseline neural network (Dense layers)

Train a more advanced CNN

Compare both models performance

Evaluate results using:

Accuracy

Confusion matrix

Misclassified examples

Discuss limitations and future improvements

Dataset: MNIST

70,000 images of handwritten digits

28×28 pixels, grayscale

10 classes (digit labels 0–9)

Split into:

60,000 training images

10,000 test images

The dataset is loaded directly from tensorflow.keras.datasets.


Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Scikit-Learn

Jupyter Notebook

Project Structure
mnist_project.ipynb        # Full notebook with code, outputs, and results
slides/                    # Presentation slides
images/                    # Saved plots, confusion matrix, misclassified digits
README.md                  # Project description

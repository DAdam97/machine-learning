# Student Depression Classification using Deep Learning

This project implements a **Deep Neural Network (DNN)** to analyze and predict student depression based on various lifestyle, academic, and demographic factors. The goal is to identify at-risk students using machine learning techniques.

The model is built using **TensorFlow/Keras** and achieves a validation accuracy of approximately **84%** with robust evaluation metrics.

## 📌 Project Overview

Mental health is a critical aspect of student life. This project utilizes a dataset containing information about study habits, sleep patterns, dietary choices, and academic pressure to classify students into two categories:
* **0:** No Depression
* **1:** Depression

The solution features an end-to-end machine learning pipeline, including advanced data preprocessing, feature engineering, and automated hyperparameter optimization.

## 🚀 Key Features

* **Deep Learning Architecture:** A custom-built neural network with **7 hidden layers**, designed to capture complex non-linear patterns in the data.
* **Hyperparameter Tuning:** Utilized **Keras Tuner (Hyperband)** to automatically search for the optimal learning rate and network structure (number of neurons).
* **Advanced Preprocessing:**
    * **Outlier Removal:** Applied `IsolationForest` to filter out statistical anomalies.
    * **Feature Engineering:** Created a custom `Pressure_Index` feature (derived from Work Pressure and CGPA) to quantify academic stress.
    * **Normalization:** Used `StandardScaler` to standardize all numerical features.
* **Regularization:** Implemented **Dropout**, **BatchNormalization**, and **L2 Regularization** to prevent overfitting and ensure model stability.

## 📊 Model Performance

The final model was evaluated on a test set using multiple metrics to ensure reliability:

* **Accuracy:** ~84%
* **AUC (Area Under Curve):** High discriminative ability (see ROC curve in the notebook).
* **Evaluation Metrics:** Precision, Recall, and F1-Score were monitored to ensure balanced classification.

## 🛠️ Tech Stack & Libraries

* **Python**
* **TensorFlow / Keras** (Deep Learning)
* **Keras Tuner** (Hyperparameter Optimization)
* **Scikit-learn** (Preprocessing & Evaluation)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)

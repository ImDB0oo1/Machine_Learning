# Income Prediction Model

This repository contains a machine learning pipeline for predicting income levels based on demographic and socioeconomic data. The model processes the Adult dataset to classify individuals into two income categories: `<=50K` and `>50K`.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Getting Started](#getting-started)
- [Future Improvements](#future-improvements)

## Overview

The project uses the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult), which contains various features such as age, education, marital status, occupation, and more. The goal is to predict whether an individual's income exceeds $50K based on these attributes.

## Key Features

- **Data Cleaning**: Handles missing values and standardizes features
- **Feature Engineering**: Includes binning for age and hours-per-week, categorical encoding, and feature transformations
- **Exploratory Data Analysis (EDA)**: Visualizes distributions and relationships between features
- **Dimensionality Reduction**: Implements PCA for visualizing data
- **Machine Learning Models**: Trains classifiers such as SVM, KNN, and Naive Bayes

## Data Preprocessing

- **Handling Missing Values**:  
  Missing values are replaced with NaN, and rows with missing data are dropped to ensure clean input.

- **Feature Encoding**:  
  Categorical variables are converted into numerical values using `LabelEncoder` to prepare the data for machine learning models.

- **Feature Transformation**:  
  Continuous variables such as age and hours-per-week are binned into discrete ranges to capture non-linear relationships.

- **Normalization**:  
  Numerical features are scaled using `StandardScaler` to standardize the data and improve model performance.

## Exploratory Data Analysis

Several visualizations are created to understand the data better:

- **Income Distribution**:  
  A count plot showing the distribution of the income categories (`<=50K` vs. `>50K`).

- **Education Levels**:  
  A visualization of the distribution of education levels, grouped for clarity.

- **Marital Status**:  
  A count plot visualizing the distribution of marital status categories.

- **PCA Visualization**:  
  Principal Component Analysis (PCA) is used to reduce dimensionality and visualize the variance captured by the first few principal components.

## Machine Learning Pipeline

The project applies the following machine learning techniques:

- **Support Vector Machines (SVM)**:  
  A classifier that separates the data by finding the hyperplane with the maximum margin. Hyperparameter tuning is applied to improve performance.

- **K-Nearest Neighbors (KNN)**:  
  A classification algorithm that assigns class labels based on the distance to the nearest neighbors. The model's performance is evaluated with different distance metrics.

- **Naive Bayes Classifier**:  
  A probabilistic classifier that's particularly efficient for categorical data. It uses Bayes' theorem to predict income categories.

## Getting Started

To run the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/income-prediction-model.git
    cd income-prediction-model
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the training and evaluation script:

    ```bash
    python train_model.py
    ```

This will:
- Load and preprocess the data
- Train multiple machine learning models
- Evaluate model performance and display results

## Future Improvements

- **Hyperparameter Tuning**:  
  Implement grid search or randomized search for further optimization of model hyperparameters.

- **Feature Engineering**:  
  Explore additional feature transformations such as one-hot encoding for categorical features or interactions between features.

- **Model Comparison**:  
  Include additional classifiers such as Decision Trees, Random Forest, and XGBoost for comparison.

- **Deployment**:  
  Implement a simple web application to deploy the trained model for real-time predictions.

---

Feel free to contribute by opening issues or submitting pull requests to improve the project.

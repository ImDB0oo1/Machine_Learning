# Comprehensive Machine Learning Projects

This repository showcases four end-to-end machine learning projects, each addressing a specific domain with unique challenges and solutions. These projects cover key aspects of data preprocessing, model training, evaluation, and visualization, demonstrating a variety of machine learning techniques and deep learning architectures.

---

## Projects Overview

### 1. **Handwritten Digit Classification**
- **Objective**: Classify handwritten digits from the MNIST dataset using Convolutional Neural Networks (CNNs).
- **Highlights**:
  - Preprocessed the dataset by normalizing pixel values and reshaping images.
  - Built a CNN architecture with convolutional layers, max pooling, and dense layers.
  - Achieved ~98% accuracy on the test set.
  - Evaluated performance using confusion matrices and classification reports.
- **Key Features**:
  - **Dataset**: MNIST dataset.
  - **Model**: CNN with ReLU activation, SGD optimizer, and softmax output.
  - **Training**: Early stopping callback to prevent overfitting.Validation accuracy tracking.
  - **Visualization**: Displayed training samples, confusion matrices, and misclassified digits.

### 2. **Transfer Learning for CIFAR-10 Classification**
- **Objective**: Classify CIFAR-10 images using a pre-trained DenseNet121 architecture.
- **Highlights**:
  - Applied transfer learning with ImageNet weights and fine-tuned a DenseNet121 model.
  - Adapted the CIFAR-10 dataset by resizing images to match DenseNet input dimensions.
  - Achieved high classification accuracy using the Adam optimizer and dropout regularization.
- **Key Features**:
  - **Dataset**: CIFAR-10 dataset.
  - **Model**: DenseNet121 with custom fully connected layers.
  - **Visualization**: Plotted training and validation loss trends, confusion matrices.

### 3. **Weather Prediction**
- **Objective**: Predict weather conditions (`RainToday`) using time-series data and deep learning models.
- **Highlights**:
  - Preprocessed data by handling missing values, encoding categorical features, and scaling numerical features.
  - Implemented LSTM and GRU models for time-series classification.
  - Balanced the dataset using SMOTE for better model performance.
  - Evaluated performance with confusion matrices, loss trends, and classification metrics.
- **Key Features**:
  - **Dataset**: Australian weather dataset.
  - **Models**: LSTM and GRU for sequential data.
  - **Visualization**: Displayed training loss trends and confusion matrices.

### 4. **Fraud Detection**
- **Objective**: Detect fraudulent transactions in application data using various machine learning techniques.
- **Highlights**:
  - Preprocessed the dataset by handling missing values, encoding categorical variables, and scaling features.
  - Balanced the dataset using SMOTE to address class imbalance.
  - Trained multiple models: Decision Tree, Neural Network, Autoencoder, and Variational Autoencoder (VAE).
  - Analyzed reconstruction errors for fraud detection and compared model performances.
- **Key Features**:
  - **Dataset**: Application data with numerical and categorical variables.
  - **Models**: Decision Tree, Neural Network, Autoencoder, Variational Autoencoder.
  - **Visualization**: Feature importance, reconstruction error plots, and confusion matrices.

---

## Key Techniques and Methods

1. **Data Preprocessing**:
   - Missing value handling using mean and mode imputation.
   - Encoding categorical variables with `OrdinalEncoder`.
   - Scaling numerical features with `MinMaxScaler`.
   - Balancing datasets using SMOTE.

2. **Machine Learning Models**:
   - Decision Trees for baseline performance and feature importance analysis.
   - Fully Connected Neural Networks for classification tasks.
   - Autoencoders for anomaly detection.

3. **Deep Learning Architectures**:
   - Convolutional Neural Networks (CNNs) for image classification.
   - Transfer Learning with DenseNet121 for image recognition.
   - LSTM and GRU for time-series forecasting.
   - Variational Autoencoders (VAEs) for anomaly detection and feature extraction.

4. **Evaluation Metrics**:
   - Classification reports with precision, recall, F1-score, and support.
   - Confusion matrices for detailed error analysis.
   - PR-AUC and ROC-AUC for threshold evaluation.

5. **Visualization**:
   - Loss trends during training and validation.
   - Feature importance plots for interpretability.
   - Reconstruction error analysis for anomaly detection.

---

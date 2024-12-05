# Clustering and Dimensionality Reduction using Multiple Algorithms

This repository provides a comprehensive machine learning project that applies multiple clustering and dimensionality reduction techniques to a dataset. The project explores various clustering methods, including K-Means, Gaussian Mixture Model (GMM), Spectral Clustering, and custom K-Means, along with dimensionality reduction using PCA (Principal Component Analysis). The goal is to demonstrate the clustering behavior and reduce high-dimensional data to a more interpretable format.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Preprocessing](#data-preprocessing)
- [Clustering Algorithms](#clustering-algorithms)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Visualizations](#visualizations)

## Overview

This project works with a dataset related to various countries and their attributes like child mortality, exports, health, inflation, income, and others. The goal is to cluster the data into groups based on these features using different clustering algorithms and to apply dimensionality reduction techniques to visualize high-dimensional data.

## Key Features

### Data Preprocessing:
- Standardization of numerical features.
- Handling missing values.
- Box plot visualizations for outlier detection.

### Clustering Algorithms:
- **K-Means**: Standard K-Means algorithm for clustering and analysis using the elbow method and silhouette score.
- **Gaussian Mixture Model (GMM)**: Fit and predict clusters using GMM.
- **Spectral Clustering**: A graph-based clustering algorithm.
- **Custom K-Means Clustering**: A custom implementation of the K-Means algorithm with visualization.

### Dimensionality Reduction:
- **PCA**: Used to reduce dimensionality while maintaining the maximum variance.
- **Incremental PCA**: An alternative PCA for handling large datasets.

### Model Evaluation:
- Silhouette Score for cluster evaluation.
- Davies-Bouldin and Calinski-Harabasz scores.

### Visualizations:
- Heatmaps for correlation.
- Box plots for outlier detection.
- Scatter plots for clustering results.
- 3D visualization of principal components after PCA.

## Data Preprocessing

The preprocessing steps involved in this project include:

- **Handling Missing Values**: Missing values are replaced with NaN, and rows containing NaN values are dropped.
- **Feature Standardization**: Features are standardized using StandardScaler to transform the data into a distribution with a mean of 0 and a standard deviation of 1.
- **Outlier Detection**: Descriptive statistics are generated to detect potential outliers in the dataset. Box plots are used to visualize the outliers.
- **Correlation Matrix**: A heatmap is plotted to visualize the correlations between different features in the dataset.

## Clustering Algorithms

The following clustering algorithms are applied to the dataset:

### 1. K-Means Clustering
- The elbow method and silhouette score are used to determine the optimal number of clusters.
- The K-Means algorithm is applied to standardized data to generate clusters.
- Visualizations of clustering results using scatter plots with different feature combinations.

### 2. Gaussian Mixture Model (GMM)
- GMM is used to cluster data into multiple Gaussian distributions, and the best number of clusters is selected using the silhouette score.
- Scatter plots are used to visualize the results.

### 3. Spectral Clustering
- Spectral clustering is applied using the nearest_neighbors affinity method.
- The optimal number of clusters is determined by evaluating different values using silhouette scores.

### 4. Custom K-Means Implementation
- A custom implementation of K-Means clustering is used to compare its performance with the built-in KMeans algorithm from sklearn.

## Dimensionality Reduction

### 1. Principal Component Analysis (PCA)
- PCA is used to reduce the dimensions of the dataset to 2 or 3 components, making it easier to visualize the data.
- The cumulative explained variance ratio is plotted to determine how many components are needed to explain the majority of the variance.

### 2. Incremental PCA
- Incremental PCA is used to handle large datasets that may not fit in memory all at once.
- The first few principal components are visualized using boxplots and scatter plots.

## Visualizations

Visualizations are a key part of this project, as they help understand the clustering results and data distributions:

- **Correlation Heatmap**: Shows correlations between different features in the dataset.
- **Box Plots**: Used to detect outliers in the dataset.
- **Scatter Plots**: Visualize clusters formed by K-Means, GMM, and Spectral Clustering.
- **3D Visualization**: After applying PCA, the first three principal components are plotted in 3D to visualize the reduced dataset.

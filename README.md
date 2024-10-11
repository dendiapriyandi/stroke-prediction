# Stroke Prediction
This repository contains a Jupyter Notebook that explores the relationship between various health habits and the likelihood of a stroke. The notebook aims to visualize this relationship and predict the probability of stroke occurrences using the best machine learning model with hyperparameter tuning.
## Problem Statement

The objective of this notebook is to visualize the relationship between various healthy and unhealthy habits and the occurrence of strokes. Additionally, it aims to predict the probability of stroke occurrences using the best-performing machine learning model and optimize it using hyperparameter tuning.
## Authors

This project was developed by [Dendi Apriyandi](https://www.linkedin.com/in/dendiapriyandi), an entry level data analyst and business intelligence.
## Dataset

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

The dataset contains health-related features that are known to impact the likelihood of stroke, including:

- Age
- Gender
- Smoking habits
- Body Mass Index (BMI)
- Average glucose level
- Presence of heart disease, etc.
## Modeling

The project experimented with various machine learning models to predict stroke occurrences. These models were fine-tuned using hyperparameter optimization techniques. The following steps were performed:

- Data preprocessing: Handling missing values, encoding categorical data, etc.
- Feature selection: Identifying the most relevant features for prediction
- Model selection: Several models were trained, including decision trees, logistic regression, and more
- Hyperparameter tuning: Grid search or random search methods were employed to optimize model performance
## Introduction

The primary goal of this project is to predict stroke occurrences based on health data.
## Results

The key insights and results from the analysis include:

Visualizations that depict the relationships between health habits and the occurrence of strokes
Prediction performance of the best model with the highest accuracy and other metrics such as precision, recall, and F1-score
## Conclusion

This project highlights the importance of lifestyle factors in predicting stroke occurrences. The best-performing model was identified and fine-tuned, providing accurate predictions and useful insights for potential stroke prevention strategies.

The best-performing model in this project was the K-Nearest Neighbors (KNN) model with the following hyperparameters:

weights: 'distance'
n_neighbors: 3
leaf_size: 35
algorithm: 'auto'
After hyperparameter tuning, the performance of the best model was as follows:

Train Accuracy: 1.0000 (indicating overfitting on the training data)
Train Recall: 1.0000
Test Accuracy: 0.9247
Test Recall: 0.1452
While the model showed high accuracy, the recall score on the test data was quite low, indicating that the model struggled to detect stroke cases effectively.
## Requirements

To run the notebook, you will need the following libraries:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

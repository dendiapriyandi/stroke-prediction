
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

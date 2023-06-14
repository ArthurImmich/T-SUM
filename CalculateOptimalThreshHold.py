from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np

# Assume you have a set of predicted probabilities and true labels for a validation dataset
y_prob = [0.5, 0.6, 0.7, 0.7]
y_true = [0, 0, 1, 1]

# Calculate precision, recall, and thresholds
_, _, thresholds = precision_recall_curve(y_true, y_prob)

# Calculate F1 score for each threshold
f1_scores = [f1_score(y_true, y_prob >= t) for t in thresholds]

# Find the threshold that gives the highest F1 score
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Apply the optimal threshold to your predicted probabilities to get binary class labels
y_pred = (y_prob >= optimal_threshold).astype(int)

print(y_pred)

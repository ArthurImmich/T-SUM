from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np

# Assume you have a set of predicted probabilities and true labels for a validation dataset
y_prob = model.predict_proba(X_val)[:, 1]
y_true = y_val

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

# Calculate F1 score for each threshold
f1_scores = [f1_score(y_true, y_prob >= t) for t in thresholds]

# Find the threshold that gives the highest F1 score
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Apply the optimal threshold to your predicted probabilities to get binary class labels
y_pred = (y_prob >= optimal_threshold).astype(int)

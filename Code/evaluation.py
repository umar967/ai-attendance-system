import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from database import load_face_encodings


RANDOM_STATE = 333
TEST_SIZE = 0.2


def load_evaluation_data():
    """Load face encodings and labels from MySQL for model evaluation."""
    known_encodings, known_names, _ = load_face_encodings()
    if not known_encodings:
        raise RuntimeError("No MySQL face encodings found for evaluation.")
    return np.array(known_encodings), np.array(known_names)


X, y = load_evaluation_data()

# Test-train splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Check samples
print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")

# Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)

# Training and Prediction
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Evaluation Metrics
print("EVALUATION METRICS")


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
# scores
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score : {f1:.4f} ({f1*100:.2f}%)")

# Confusion Matrix

print("CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y),
            cmap="Blues")

plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Face Recognition Model", fontsize=14)
plt.tight_layout()
plt.show()

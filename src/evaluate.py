import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the balanced dataset
df = pd.read_csv("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/data/balanced_bug_reports.csv")

# Step 2: Load the trained SVM model and vectorizer
model = joblib.load("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/results/svm_model.pkl")
vectorizer = joblib.load("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/results/tfidf_vectorizer.pkl")

# Step 3: Prepare test data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['category'], test_size=0.2, random_state=42
)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Make predictions and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save classification report
with open("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/results/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# Step 5: Create and save confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/results/confusion_matrix.png")
plt.show()

 

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load the balanced dataset
df = pd.read_csv("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/data/balanced_bug_reports.csv")

# Step 2: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['category'], test_size=0.2, random_state=42
)

# Step 3: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Step 4: Train an SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Step 5: Save the model and vectorizer
joblib.dump(svm_model, "C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/results/svm_model.pkl")
joblib.dump(vectorizer, "C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/results/tfidf_vectorizer.pkl")

print("✅ SVM model trained and saved.")




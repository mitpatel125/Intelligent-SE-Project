import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords and tokenizer data (only the first time)
nltk.download('stopwords')
nltk.download('punkt')

# Text cleaning function
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load raw bug report data
df = pd.read_csv("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/data/bug_reports.csv")

# Clean each report
df['cleaned_text'] = df['report_text'].apply(preprocess_text)

# Save the cleaned dataset
df.to_csv("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/data/cleaned_bug_reports.csv", index=False)
print("✅ Preprocessing complete. Cleaned dataset saved.")


 

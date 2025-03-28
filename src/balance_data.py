import pandas as pd
from sklearn.utils import resample

# Load cleaned dataset
df = pd.read_csv("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/data/cleaned_bug_reports.csv")

# Separate by category
df_bug = df[df['category'] == "Bug"]
df_feature = df[df['category'] == "Feature Request"]
df_enhancement = df[df['category'] == "Enhancement"]

# Find the smallest class size
min_samples = min(len(df_bug), len(df_feature), len(df_enhancement))

# Undersample each class to the same size
df_bug = resample(df_bug, replace=False, n_samples=min_samples, random_state=42)
df_feature = resample(df_feature, replace=False, n_samples=min_samples, random_state=42)
df_enhancement = resample(df_enhancement, replace=False, n_samples=min_samples, random_state=42)

# Combine and save
df_balanced = pd.concat([df_bug, df_feature, df_enhancement])
df_balanced.to_csv("C:/Users/mitpa/OneDrive/Desktop/bug-report-classification/data/balanced_bug_reports.csv", index=False)

print("✅ Dataset balanced and saved.")

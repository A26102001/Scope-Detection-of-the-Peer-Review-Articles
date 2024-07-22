import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load training and validation data
train_df = pd.read_csv(r"./train.csv", header=None, names=["abs_text", "label_text"])
val_df = pd.read_csv(r"./validation.csv", header=None, names=["abs_text", "label_text"])

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing to training data
train_df['cleaned_abstract'] = train_df['abs_text'].apply(preprocess_text)

# Split validation data into validation and test sets
val_df['cleaned_abstract'] = val_df['abs_text'].apply(preprocess_text)
X_val, X_test, y_val, y_test = train_test_split(val_df['cleaned_abstract'], val_df['label_text'], test_size=0.2, random_state=42)

# Extract features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_df['cleaned_abstract'])
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga', verbose=1),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Voting Classifier": VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, solver='saga')),
            ('nb', MultinomialNB()),
            ('svc', SVC(probability=True)),
            ('rf', RandomForestClassifier()),
            ('gb', GradientBoostingClassifier())
        ],
        voting='soft'
    )
}

# Train and evaluate models
results = []
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_tfidf, train_df['label_text'])
    
    # Predict on validation data
    y_val_pred = model.predict(X_val_tfidf)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    # Predict on test data
    y_test_pred = model.predict(X_test_tfidf)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Classification report for test set
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Save results
    results.append({
        "Model": model_name,
        "Validation F1": val_f1,
        "Test F1": test_f1,
        "Classification Report": report
    })
    print(f"{model_name} - Validation F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")

# Create a DataFrame to compare results
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Save the detailed classification reports for each model
for result in results:
    model_name = result['Model']
    report_df = pd.DataFrame(result['Classification Report']).transpose()
    report_df.to_csv(f'{model_name}_classification_report.csv', index=True)

# Save the overall results to a CSV file
results_df.to_csv('MachineLearning_comparison_results.csv', index=False)

# Display the overall comparison DataFrame
print(results_df)

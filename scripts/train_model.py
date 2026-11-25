
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

# Load data
print("1-Loading data...")
train_df = pd.read_csv('data/train_clean.csv')
test_df = pd.read_csv('data/test_clean.csv')
#texti al mood olarak çıkar
X_train = train_df['text']
y_train = train_df['mood']
X_test = test_df['text']
y_test = test_df['mood']

#train sample testi
print(f"   Train: {len(train_df)} samples")


print("\n2-TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=3000,      
    ngram_range=(1, 2),     
    min_df=2              
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Feature count: {X_train_vec.shape[1]} features")

# model egitim
print("\n3- Training Logistic Regression model")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
     # bazı labellar çok az, onu dengelemem gerek
    class_weight='balanced' 
)

model.fit(X_train_vec, y_train)
print("model trtained")


print("\n4-Evaluating model...")
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")

print("\nDetailed results per mood:")
print(classification_report(y_test, y_pred, target_names=model.classes_))


print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Predicted →")
print(f"{'':10s}", end='')
for cls in model.classes_:
    print(f"{cls:>8s}", end='')
print()
for i, cls in enumerate(model.classes_):
    print(f"{cls:10s}", end='')
    for val in cm[i]:
        print(f"{val:>8d}", end='')
    print()

max_probas = y_proba.max(axis=1)
print(f"\nConfidence statistics:")
print(f"   Average confidence: {max_probas.mean():.2%}")
print(f"   Low confidence samples (<45%): {(max_probas < 0.45).sum()} ({(max_probas < 0.45).mean()*100:.1f}%)")

print("\n Saving model.")
joblib.dump({
    'vectorizer': vectorizer,
    'model': model,
    'classes': model.classes_.tolist()
}, 'models/mood_model.joblib')

print("Model saved yey")


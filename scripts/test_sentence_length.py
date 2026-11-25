import joblib
import pandas as pd

data = joblib.load('models/mood_model.joblib')
vectorizer = data['vectorizer']
model = data['model']

tests = [
    ("I love you so much", "love"),
    ("I'm feeling great today", "happy"),
    ("I feel so sad and alone", "sad"),
    
    ("I love you. You're amazing.", "love"),
    ("I'm happy today. Everything is great.", "happy"),
    ("I feel sad. Nothing works.", "sad"),
    
    ("I love you. You're amazing. Best ever. So grateful.", "love"),
    ("I'm happy. Great day. Everything perfect. Life is good.", "happy"),
    ("I'm sad. Everything bad. Nothing works. Terrible day.", "sad"),
]

print("Sentence Count vs Confidence:\n")
for text, expected in tests:
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    if sentence_count == 0:
        sentence_count = 1
    
    vec = vectorizer.transform([text])
    mood = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0].max()
    
    match = "✓" if mood == expected else "✗"
    print(f"{sentence_count} sent | {confidence:.0%} | {match} | {text[:40]}...")


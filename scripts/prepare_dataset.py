
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("dair-ai/emotion")

# Dataframeler
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])


# label moods
mood_map = {0: 'sad', 1: 'happy', 2: 'love', 3: 'angry', 5: 'neutral'}

# removed fear mood
train_df = train_df[train_df['label'] != 4]
test_df = test_df[test_df['label'] != 4]

train_df['mood'] = train_df['label'].map(mood_map)
test_df['mood'] = test_df['label'].map(mood_map)

# Keep only text and mood
train_df = train_df[['text', 'mood']]
test_df = test_df[['text', 'mood']]

# Save
train_df.to_csv('data/train_clean.csv', index=False)
test_df.to_csv('data/test_clean.csv', index=False)

# print(f"\nSaved {len(train_df)} train samples")
# print(f"Saved {len(test_df)} test samples")
print("\nMood counts:")
print(train_df['mood'].value_counts())

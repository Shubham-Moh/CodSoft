import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
nltk.download('punkt')

train_file_path = "datasets/Genre Classification Dataset/train_data.txt"
test_file_path = "datasets/Genre Classification Dataset/test_data.txt"
solution_file_path = "datasets/Genre Classification Dataset/test_data_solution.txt"

for file_path in [train_file_path, test_file_path, solution_file_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

def load_dataset(file_path, is_train=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if is_train and len(parts) == 4:
                data.append((parts[2], parts[3]))  
            elif not is_train and len(parts) == 3:
                data.append((parts[0], parts[2]))  
    df = pd.DataFrame(data, columns=['genre', 'description']) if is_train else pd.DataFrame(data, columns=['ID', 'description'])
    return df

df = load_dataset(train_file_path, is_train=True)
print("Train dataset loaded successfully")

df = df.sample(n=1000, random_state=42)  


df.dropna(subset=['description'], inplace=True)


def clean_text(text):
    if isinstance(text, float): 
        return ""
    text = text.lower()  
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text) 
    words = word_tokenize(text) 
    words = [word for word in words if word not in stopwords.words('english')] 
    return " ".join(words)

df['cleaned_description'] = df['description'].apply(clean_text)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_description'])
y = df['genre']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


test_df = load_dataset(test_file_path, is_train=False)
test_df = test_df.sample(n=100, random_state=42)  
print("Test dataset loaded successfully")


test_df['cleaned_description'] = test_df['description'].apply(clean_text)


X_test_final = vectorizer.transform(test_df['cleaned_description'])

test_df['predicted_genre'] = model.predict(X_test_final)


def load_solution(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                data.append((parts[0], parts[2]))  
    return pd.DataFrame(data, columns=['ID', 'genre'])

solution_df = load_solution(solution_file_path)
solution_df = solution_df.sample(n=100, random_state=42)  


merged_df = test_df[['ID', 'description', 'predicted_genre']].merge(solution_df, on='ID', how='left')


test_accuracy = accuracy_score(merged_df['genre'], merged_df['predicted_genre'])
print(f'Test Accuracy: {test_accuracy:.4f}')
print("Test Classification Report:")
print(classification_report(merged_df['genre'], merged_df['predicted_genre'], zero_division=0))


print("\nSample Predictions:")
print(merged_df[['description', 'genre', 'predicted_genre']].head(5))

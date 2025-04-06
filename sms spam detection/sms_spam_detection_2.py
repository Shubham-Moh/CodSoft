import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("datasets/SMS Spam Detection Dataset/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['cleaned_message'] = df['message'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_message'], df['label'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print("Evaluation on Test Set")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))


print("\nSample Predictions on Test Data:")
test_sample = pd.DataFrame({'message': X_test, 'actual': y_test, 'predicted': y_pred})
sample = test_sample.sample(20, random_state=40)

for i, row in sample.iterrows():
    print(f"\nMessage: {row['message']}")
    print(f"Actual: {'Spam' if row['actual'] else 'Ham'}")
    print(f"Predicted: {'Spam' if row['predicted'] else 'Ham'}")

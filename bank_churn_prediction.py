import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("datasets/Bank Churn Prediction/Churn_Modelling.csv")  

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)


label_encoders = {}
for col in ['Geography', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=['Exited']) 
y = df['Exited'] 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def predict_churn(new_data):
    new_data_df = pd.DataFrame([new_data], columns=X.columns)
    
    
    for col in ['Geography', 'Gender']:
        new_data_df[col] = label_encoders[col].transform([new_data_df[col][0]])
    
    new_data_scaled = scaler.transform(new_data_df)
    

    best_model = models["Gradient Boosting"]
    prediction = best_model.predict(new_data_scaled)
    
    return "Churn" if prediction[0] == 1 else "No Churn"

example_customer = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 45,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 80000
}

print("\nExample Customer Prediction:")
print(predict_churn(example_customer))

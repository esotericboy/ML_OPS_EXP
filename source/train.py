import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = datasets.load_iris()
    
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
    
print("--- Dataset Sample ---")
print(X.head())
print("\nTarget classes:", iris.target_names)
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_test_split=0.2, random_state=42, stratify=y
)
    
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
    
y_pred = model.predict(X_test)
    
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
    
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
    
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

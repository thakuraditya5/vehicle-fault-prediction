import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

FILE_PATH = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_Empire_AC_UK07PA6111_bcs.xlsx"

def train_bcs_model():
    print(f"Loading full dataset from {FILE_PATH}...")
    # Read full file
    df = pd.read_excel(FILE_PATH)
    
    print(f"Dataset Shape: {df.shape}")
    
    # Check Target Distribution
    print("Target Distribution:")
    print(df['BCSfault'].value_counts())
    
    if len(df['BCSfault'].unique()) < 2:
        print("\nCRITICAL ERROR: The target column 'BCSfault' has only ONE value.")
        print("Model cannot learn to differentiate valid vs invalid if there are no examples of both.")
        return

    # Preprocessing
    # Drop timestamp, target, LEAKY features, and SYMPTOM features (power/voltage drops)
    leaky_cols = ['BCSfault', 'Server_Time', 'BCScompressorerror', 'A_Fault_Rank', 'B_Fault_Rank',
                  'BCScompressorinputvoltage', 'BCScompressorinputpower', 'BCScompressorinputcurrent', 'BCSFlag']
    X = df.drop(leaky_cols, axis=1, errors='ignore')
    y = df['BCSfault']
    
    # Handle NaN
    X = X.fillna(0) 
    
    print("\nFeatures used:", X.columns.tolist())
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Random Forest...")
    # class_weight='balanced' helps if faults are rare
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    if not os.path.exists("images"):
        os.makedirs("images")
        
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("BCS Fault - Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("images/bcs_feature_importance.png")
    plt.close()
    
    print("Feature importance saved to 'images/bcs_feature_importance.png'")

    # Save Model
    joblib.dump(model, "bcs_fault_model.pkl")
    print("Model saved to 'bcs_fault_model.pkl'")

if __name__ == "__main__":
    train_bcs_model()

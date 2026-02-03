import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

FILE_PATH = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_MBMT_NAC_MH04LQ9368_bcs.xlsx"

def train_mbmt_model():
    print(f"Loading MBMT dataset from {FILE_PATH}...")
    df = pd.read_excel(FILE_PATH)
    
    print(f"Dataset Shape: {df.shape}")
    
    # Check Target Distribution
    print("\nTarget Distribution:")
    print(df['BCSfault'].value_counts())
    print(f"Fault Rate: {df['BCSfault'].sum()/len(df)*100:.2f}%")
    
    # Preprocessing - same approach
    leaky_cols = ['BCSfault', 'Server_Time', 'BCScompressorerror', 'A_Fault_Rank', 'B_Fault_Rank',
                  'BCScompressorinputvoltage', 'BCScompressorinputpower', 'BCScompressorinputcurrent', 'BCSFlag']
    X = df.drop(leaky_cols, axis=1, errors='ignore')
    y = df['BCSfault']
    
    # Handle NaN
    X = X.fillna(0) 
    
    print("\nFeatures used:", X.columns.tolist())
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("\nTraining Random Forest on MBMT data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"MBMT MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    if not os.path.exists("images"):
        os.makedirs("images")
        
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("MBMT Model - Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("images/mbmt_feature_importance.png")
    plt.close()
    
    print("\nFeature importance saved to 'images/mbmt_feature_importance.png'")

    # Save Model
    joblib.dump(model, "bcs_fault_model_mbmt.pkl")
    print("Model saved to 'bcs_fault_model_mbmt.pkl'")
    
    # Detailed breakdown
    print(f"\n{'='*60}")
    print("DETAILED PERFORMANCE")
    print(f"{'='*60}")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    total_faults = y_test.sum()
    detected_faults = ((y_test == 1) & (y_pred == 1)).sum()
    print(f"\nFault Detection:")
    print(f"  Total Faults in Test Set: {total_faults}")
    print(f"  Correctly Detected: {detected_faults}")
    print(f"  Missed: {total_faults - detected_faults}")
    print(f"  Detection Rate (Recall): {detected_faults/total_faults*100:.2f}%")

if __name__ == "__main__":
    train_mbmt_model()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# File paths
FILE1 = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_Empire_AC_UK07PA6111_bcs.xlsx"
FILE2 = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_Empire_AC_UK07PA7011_bcs.xlsx"

def train_combined_model():
    print("Loading datasets...")
    df1 = pd.read_excel(FILE1)
    df2 = pd.read_excel(FILE2)
    
    # Add source identifier
    df1['source'] = 'PA6111'
    df2['source'] = 'PA7011'
    
    # Combine datasets
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    print(f"\nCombined Dataset Shape (Before Cleaning): {df_combined.shape}")
    
    # CLEAN DATA: Remove sensor failure outliers (-40°C is a sensor error code)
    print("\nCleaning sensor failure outliers...")
    initial_count = len(df_combined)
    
    # Filter out rows where thermistors show sensor failure
    df_combined = df_combined[
        (df_combined['BCSThermistor1'] > -35) &  # Allow some tolerance
        (df_combined['BCSThermistor2'] > -35)
    ]
    
    removed_count = initial_count - len(df_combined)
    print(f"Removed {removed_count:,} records with sensor failures ({removed_count/initial_count*100:.2f}%)")
    print(f"Cleaned Dataset Shape: {df_combined.shape}")
    
    print("\nTarget Distribution:")
    print(df_combined['BCSfault'].value_counts())
    print(f"\nTotal Faults: {df_combined['BCSfault'].sum()} ({df_combined['BCSfault'].sum()/len(df_combined)*100:.3f}%)")
    
    # Preprocessing - same as before
    leaky_cols = ['BCSfault', 'Server_Time', 'BCScompressorerror', 'A_Fault_Rank', 'B_Fault_Rank',
                  'BCScompressorinputvoltage', 'BCScompressorinputpower', 'BCScompressorinputcurrent', 
                  'BCSFlag', 'source']
    X = df_combined.drop(leaky_cols, axis=1, errors='ignore')
    y = df_combined['BCSfault']
    
    # Handle NaN
    X = X.fillna(0)
    
    print("\nFeatures used:", X.columns.tolist())
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("\nTraining Random Forest on Combined Data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"COMBINED MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    if not os.path.exists("images"):
        os.makedirs("images")
        
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Combined Model - Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("images/combined_feature_importance.png")
    plt.close()
    
    print("\nFeature importance saved to 'images/combined_feature_importance.png'")
    
    # Save Model
    joblib.dump(model, "bcs_fault_model_combined.pkl")
    print("Model saved to 'bcs_fault_model_combined.pkl'")
    
    # Test on each dataset separately
    print(f"\n{'='*60}")
    print("TESTING ON INDIVIDUAL DATASETS")
    print(f"{'='*60}")
    
    # Test on PA6111
    X_test_6111 = df1.drop(leaky_cols, axis=1, errors='ignore').fillna(0)
    y_test_6111 = df1['BCSfault']
    y_pred_6111 = model.predict(X_test_6111)
    acc_6111 = accuracy_score(y_test_6111, y_pred_6111)
    print(f"\nPA6111 Accuracy: {acc_6111:.4f} ({acc_6111*100:.2f}%)")
    
    # Test on PA7011
    X_test_7011 = df2.drop(leaky_cols, axis=1, errors='ignore').fillna(0)
    y_test_7011 = df2['BCSfault']
    y_pred_7011 = model.predict(X_test_7011)
    acc_7011 = accuracy_score(y_test_7011, y_pred_7011)
    print(f"PA7011 Accuracy: {acc_7011:.4f} ({acc_7011*100:.2f}%)")

if __name__ == "__main__":
    train_combined_model()

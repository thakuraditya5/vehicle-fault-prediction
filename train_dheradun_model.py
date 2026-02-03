import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

FILE_PATH = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_DHERADUN_AC_UK07PA5910_bcs.xlsx"

def train_dheradun_model():
    print(f"Loading DHERADUN dataset from {FILE_PATH}...")
    df = pd.read_excel(FILE_PATH)
    
    print(f"Dataset Shape: {df.shape}")
    
    # Check Target Distribution
    print("\nTarget Distribution:")
    print(df['BCSfault'].value_counts())
    print(f"Fault Rate: {df['BCSfault'].sum()/len(df)*100:.2f}%")
    
    # Preprocessing
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
    print("\nTraining Random Forest on DHERADUN data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"DHERADUN MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Detailed breakdown
    total_faults = y_test.sum()
    detected_faults = ((y_test == 1) & (y_pred == 1)).sum()
    false_positives = ((y == 0) & (y_pred == 1)).sum()
    
    print(f"\n{'='*60}")
    print("DETAILED PERFORMANCE")
    print(f"{'='*60}")
    print(f"Total Faults in Test Set: {total_faults}")
    print(f"Correctly Detected: {detected_faults}")
    print(f"Missed: {total_faults - detected_faults}")
    print(f"Detection Rate (Recall): {detected_faults/total_faults*100:.2f}%")

    # Save Model
    joblib.dump(model, "bcs_fault_model_dheradun.pkl")
    print("\nModel saved to 'bcs_fault_model_dheradun.pkl'")

if __name__ == "__main__":
    train_dheradun_model()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# All MBMT buses
buses = [
    {
        'name': 'MH04LQ9368',
        'path': r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_MBMT_NAC_MH04LQ9368_bcs.xlsx"
    },
    {
        'name': 'MH04LQ9367',
        'path': r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_MBMT_NAC_MH04LQ9367_bcs.xlsx"
    },
    {
        'name': 'MH04LQ9866',
        'path': r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_MBMT_NAC_MH04LQ9866_bcs.xlsx"
    }
]

results = []

for bus_info in buses:
    bus_name = bus_info['name']
    file_path = bus_info['path']
    
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL FOR BUS: {bus_name}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading {file_path}...")
    df = pd.read_excel(file_path)
    
    print(f"Dataset Shape: {df.shape}")
    fault_rate = df['BCSfault'].sum() / len(df) * 100
    print(f"Fault Rate: {fault_rate:.2f}% ({df['BCSfault'].sum()} faults)")
    
    # Preprocessing
    leaky_cols = ['BCSfault', 'Server_Time', 'BCScompressorerror', 'A_Fault_Rank', 'B_Fault_Rank',
                  'BCScompressorinputvoltage', 'BCScompressorinputpower', 'BCScompressorinputcurrent', 'BCSFlag']
    X = df.drop(leaky_cols, axis=1, errors='ignore')
    y = df['BCSfault']
    X = X.fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate recall
    total_faults = y_test.sum()
    detected_faults = ((y_test == 1) & (y_pred == 1)).sum()
    recall = detected_faults / total_faults * 100 if total_faults > 0 else 0
    
    print(f"\nACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"RECALL (Fault Detection): {recall:.2f}%")
    print(f"Detected {detected_faults} out of {total_faults} faults")
    
    # Save model
    model_filename = f"bcs_fault_model_{bus_name}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to '{model_filename}'")
    
    results.append({
        'Bus': bus_name,
        'Fault_Rate': fault_rate,
        'Accuracy': accuracy * 100,
        'Recall': recall,
        'Total_Faults': total_faults,
        'Detected': detected_faults
    })

# Summary
print(f"\n{'='*70}")
print("SUMMARY: INDIVIDUAL BUS MODELS")
print(f"{'='*70}")
print(f"\n{'Bus':<15} {'Fault Rate':<12} {'Accuracy':<12} {'Recall':<12}")
print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12}")
for r in results:
    print(f"{r['Bus']:<15} {r['Fault_Rate']:>10.2f}% {r['Accuracy']:>10.2f}% {r['Recall']:>10.2f}%")

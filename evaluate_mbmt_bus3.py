import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the trained MBMT model
print("Loading MBMT model...")
model = joblib.load("bcs_fault_model_mbmt.pkl")

# Load new MBMT bus dataset
NEW_FILE = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_MBMT_NAC_MH04LQ9866_bcs.xlsx"
print(f"Loading dataset from {NEW_FILE}...")
df = pd.read_excel(NEW_FILE)

print(f"\nDataset Shape: {df.shape}")
print("\nTarget Distribution:")
print(df['BCSfault'].value_counts())
print(f"Fault Rate: {df['BCSfault'].sum()/len(df)*100:.2f}%")

# Preprocess - same as training
leaky_cols = ['BCSfault', 'Server_Time', 'BCScompressorerror', 'A_Fault_Rank', 'B_Fault_Rank',
              'BCScompressorinputvoltage', 'BCScompressorinputpower', 'BCScompressorinputcurrent', 'BCSFlag']
X = df.drop(leaky_cols, axis=1, errors='ignore')
y = df['BCSfault']

# Handle NaN
X = X.fillna(0)

print("\nFeatures used:", X.columns.tolist())

# Predict
print("\nEvaluating MBMT model on MH04LQ9866...")
y_pred = model.predict(X)

# Report accuracy
accuracy = accuracy_score(y, y_pred)
print(f"\n{'='*60}")
print(f"ACCURACY ON MBMT BUS (MH04LQ9866): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

print("\nDetailed Classification Report:")
print(classification_report(y, y_pred, zero_division=0))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y, y_pred)
print(cm)

# Detailed breakdown
total_faults = y.sum()
detected_faults = ((y == 1) & (y_pred == 1)).sum()
false_positives = ((y == 0) & (y_pred == 1)).sum()

print(f"\n{'='*60}")
print("DETAILED BREAKDOWN")
print(f"{'='*60}")
print(f"Total Faults: {total_faults:,}")
print(f"Correctly Detected: {detected_faults:,}")
print(f"Missed: {total_faults - detected_faults:,}")
print(f"False Alarms: {false_positives:,}")
if total_faults > 0:
    print(f"Detection Rate (Recall): {detected_faults/total_faults*100:.2f}%")

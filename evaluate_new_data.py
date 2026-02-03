import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
print("Loading trained model...")
model = joblib.load("bcs_fault_model.pkl")

# Load new dataset
NEW_FILE = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_Empire_AC_UK07PA7011_bcs.xlsx"
print(f"Loading new dataset from {NEW_FILE}...")
df = pd.read_excel(NEW_FILE)

print(f"Dataset Shape: {df.shape}")
print("\nTarget Distribution:")
print(df['BCSfault'].value_counts())

# Preprocess - same as training
leaky_cols = ['BCSfault', 'Server_Time', 'BCScompressorerror', 'A_Fault_Rank', 'B_Fault_Rank',
              'BCScompressorinputvoltage', 'BCScompressorinputpower', 'BCScompressorinputcurrent', 'BCSFlag']
X = df.drop(leaky_cols, axis=1, errors='ignore')
y = df['BCSfault']

# Handle NaN
X = X.fillna(0)

print("\nFeatures used:", X.columns.tolist())

# Predict
print("\nEvaluating model on new data...")
y_pred = model.predict(X)

# Report accuracy
accuracy = accuracy_score(y, y_pred)
print(f"\n{'='*50}")
print(f"ACCURACY ON NEW DATA: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")

print("\nDetailed Classification Report:")
print(classification_report(y, y_pred))

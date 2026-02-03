import joblib
import pandas as pd
import numpy as np

def check_feature_importance():
    try:
        model = joblib.load("bcs_fault_model.pkl")
        
        # We need to know feature names. 
        # Since sklearn models don't always store them safely in valid versions, 
        # I'll rely on the training script's print output or re-infer them.
        # But let's just use the known list from the last run:
        feature_names = ['AMaxCellTemp', 'BMaxCellTemp', 'BCSFlag', 'BCSThermistor1', 
                         'BCSThermistor2', 'BCScompressorinputpower', 
                         'BCScompressorinputvoltage', 'BCScompressorinputcurrent']
        
        importances = model.feature_importances_
        
        # Sort
        indices = np.argsort(importances)[::-1]
        
        print("\n--- Top Features ---")
        for f in range(len(feature_names)):
            print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_feature_importance()

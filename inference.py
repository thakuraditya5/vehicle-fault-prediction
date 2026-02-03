import joblib
import pandas as pd
import sys

def predict_fault(rpm, temp, vibration, oil_pressure, mileage):
    # Load model
    try:
        model = joblib.load("vehicle_fault_model.pkl")
    except FileNotFoundError:
        print("Error: Model file 'vehicle_fault_model.pkl' not found. Run train_model.py first.")
        return

    # Prepare input dataframe (must match training feature order)
    # Training columns were: rpm, engine_temperature, vibration, oil_pressure, mileage
    input_data = pd.DataFrame([{
        "rpm": rpm,
        "engine_temperature": temp,
        "vibration": vibration,
        "oil_pressure": oil_pressure,
        "mileage": mileage
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    print("\n--- Prediction Result ---")
    print(f"Input: RPM={rpm}, Temp={temp}°C, Vib={vibration}m/s², Oil={oil_pressure}psi, Miles={mileage}")
    print(f">> Predicted Status: {prediction.upper()}")
    
    print("\nConfidence Scores:")
    for cls, prob in zip(classes, probabilities):
        if prob > 0.01: # Only show significant probs
            print(f"  {cls}: {prob*100:.1f}%")

if __name__ == "__main__":
    print("Vehicle Fault Predictor")
    print("-----------------------")
    
    if len(sys.argv) > 1:
        # manual input mode via args if needed, or just demo
        pass
    
    # Interactive mode
    try:
        print("Enter sensor readings (or press Enter to use default test values):")
        
        r = input("RPM (default 3000): ")
        rpm = int(r) if r else 3000
        
        t = input("Engine Temperature (default 90): ")
        temp = float(t) if t else 90.0
        
        v = input("Vibration (default 2.0): ")
        vib = float(v) if v else 2.0
        
        o = input("Oil Pressure (default 40): ")
        oil = float(o) if o else 40.0
        
        m = input("Mileage (default 50000): ")
        miles = int(m) if m else 50000
        
        predict_fault(rpm, temp, vib, oil, miles)
        
    except ValueError:
        print("Invalid input. Please enter numbers.")

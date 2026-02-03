import pandas as pd
import numpy as np
import random

def generate_vehicle_data(num_samples=2000):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for _ in range(num_samples):
        # Base random values
        rpm = np.random.randint(1000, 6000)
        mileage = np.random.randint(1000, 200000)
        
        # Determine fault type based on probabilistic rules
        # Default state
        temp = np.random.normal(90, 10)  # Normal operating temp
        vibration = np.random.normal(2, 0.5) # Normal vibration
        oil_pressure = np.random.normal(40, 5) # Normal oil pressure
        fault_type = "Normal"
        
        # Inject faults
        weather_condition = random.choice(['Sunny', 'Rainy', 'Snowy'])
        
        chance = random.random()
        
        if chance < 0.05: # Overheat
            temp = np.random.normal(115, 10)
            fault_type = "Overheat"
        elif chance < 0.10: # Low Oil Pressure
            oil_pressure = np.random.normal(20, 5)
            fault_type = "Low Oil Pressure"
        elif chance < 0.15: # High Vibration
            vibration = np.random.normal(8, 2)
            fault_type = "Excessive Vibration"
        elif chance < 0.18 and rpm > 4500: # High RPM Stress
            temp = np.random.normal(110, 5)
            vibration = np.random.normal(5, 1)
            fault_type = "High RPM Stress"
            
        # Add NOISE Features (Irrelevant)
        radio_vol = np.random.randint(0, 100)
        driver_age = np.random.randint(18, 80)
        paint_color = random.choice([0, 1, 2, 3]) # Encoded: Red, Blue, Green, Black
            
        data.append({
            "rpm": rpm,
            "engine_temperature": round(temp, 2),
            "vibration": round(vibration, 2),
            "oil_pressure": round(oil_pressure, 2),
            "mileage": mileage,
            "radio_volume": radio_vol,
            "driver_age": driver_age,
            "paint_color": paint_color,
            "fault_type": fault_type
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic vehicle data...")
    df = generate_vehicle_data(2000)
    
    output_file = "vehicle_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Data generated successfully: {output_file}")
    print(df['fault_type'].value_counts())

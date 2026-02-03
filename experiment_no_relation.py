import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def run_garbage_experiment():
    print("WARNING: Generating completely RANDOM data with NO relationships...")
    
    # 1. Generate Garbage Data
    n_samples = 2000
    
    # Random Features (Inputs)
    df = pd.DataFrame({
        "rpm": np.random.randint(1000, 6000, n_samples),
        "engine_temperature": np.random.normal(90, 20, n_samples),
        "vibration": np.random.normal(2, 5, n_samples),
        "oil_pressure": np.random.normal(40, 10, n_samples),
        "mileage": np.random.randint(1000, 200000, n_samples)
    })
    
    # Random Target (Outputs) - No logic connecting this to inputs!
    possible_faults = ["Normal", "Overheat", "Low Oil Pressure", "Excessive Vibration", "High RPM Stress"]
    df["fault_type"] = np.random.choice(possible_faults, n_samples)
    
    print("Data Preview:\n", df.head())
    
    # 2. Train Model
    X = df.drop("fault_type", axis=1)
    y = df["fault_type"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Model on Random Data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("(Note: Since there are 5 classes, random guessing would get ~0.20 or 20%)")
    
    if accuracy < 0.25:
        print("\nCONCLUSION: The model failed to learn because there were no patterns to find.")
        print("This answers your question: If there is no relation, you CANNOT make a predictive model.")
    else:
        print("\nCONCLUSION: The model found some spurious correlations (noise), but performs poorly.")

    # 4. Viz
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"Confusion Matrix (Random Data)\nAccuracy: {accuracy:.2%}")
    plt.savefig("random_data_confusion.png")
    print("\nConfusion matrix saved to 'random_data_confusion.png'")

if __name__ == "__main__":
    run_garbage_experiment()

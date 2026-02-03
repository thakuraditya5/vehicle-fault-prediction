# Predictive Maintenance & Fault Detection

This repository contains Machine Learning tools for predicting vehicle and machinery faults. It includes synthetic data generation for educational purposes and a real-world application on BCS (Battery Cooling System) data.

## 📂 Project Structure

- `generate_data.py`: Creates synthetic vehicle sensor data (RPM, Temp, etc.) with injected faults.
- `train_model.py`: Trains a Random Forest classifier on the synthetic data.
- `inference.py`: Interactive CLI tool to test the model with manual inputs.
- `experiment_no_relation.py`: Educational script proving ML fails on random/uncorrelated data.
- `train_bcs_model.py`: **Real-world Application** - Trains a model on BCS Excel datasets.
- `images/`: Contains generated visualizations (Confusion Matrix, Feature Importance, Decision Trees).

## 🚀 Quick Start

### 1. Synthetic Demo
Run the pipeline to generate data and train the model:
```bash
python generate_data.py
python train_model.py
```
This will achieve ~96% accuracy and save plots in `images/`.

### 2. Real World BCS Analysis
To analyze the provided BCS Excel file:
```bash
python train_bcs_model.py
```

## 📊 BCS Fault Analysis Findings

We applied the Random Forest model to the `9M_Empire_AC_UK07PA6111_bcs.xlsx` dataset. Here are the key observations:

### Observation 1: The "Symptom" Trap
Initially, the model achieved **100% Accuracy**.
Upon investigation, we found the top predictors were:
1.  `BCScompressorinputvoltage`
2.  `BCScompressorinputpower`

**Insight**: The model wasn't predicting the *cause* of the failure; it was detecting that the machine had *already turned off* (Zero Volts/Power). This is useful for status monitoring but not for prediction.

### Observation 2: Root Cause Identification
We removed the "Symptom" columns (Voltage, Power, Current, Error Flags) and retrained the model using only **Thermal Sensors**.

**Results:**
-   **New Accuracy**: 98%
-   **Top Predictor**: `AMaxCellTemp` (Unit A Max Cell Temperature)

**Conclusion**: The root driver of the faults is **Overheating**. The model can now successfully predict a fault *based on temperature rise* before the system actually shuts down. 

## 🧠 Educational Experiments

### "Can ML find hidden relations?"
We added random "Noise" columns (`Radio Volume`, `Driver Age`) to the dataset.
-   **Result**: The model correctly assigned **0% Importance** to these columns.
-   **Takeaway**: Random Forest algorithms are effective at filtering out irrelevant data automatically.

### "What if there is no relation?"
We trained a model on completely random numbers.
-   **Result**: Accuracy dropping to ~20% (Random Guessing).
-   **Takeaway**: ML cannot create knowledge from noise. If there is no correlation in the physics, there is no prediction.

## 🛠 Dependencies
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   openpyxl (for reading Excel)

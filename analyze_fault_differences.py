import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
FILE1 = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_Empire_AC_UK07PA6111_bcs.xlsx"
FILE2 = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_Empire_AC_UK07PA7011_bcs.xlsx"

def analyze_fault_differences():
    print("Loading datasets...")
    df1 = pd.read_excel(FILE1)
    df2 = pd.read_excel(FILE2)
    
    # Filter to only fault cases
    faults1 = df1[df1['BCSfault'] == 1]
    faults2 = df2[df2['BCSfault'] == 1]
    
    print(f"\n{'='*60}")
    print(f"DATASET COMPARISON")
    print(f"{'='*60}")
    print(f"\nFile 1 (PA6111): {len(faults1)} faults out of {len(df1)} records ({len(faults1)/len(df1)*100:.2f}%)")
    print(f"File 2 (PA7011): {len(faults2)} faults out of {len(df2)} records ({len(faults2)/len(df2)*100:.2f}%)")
    
    # Compare temperature distributions during faults
    features = ['AMaxCellTemp', 'BMaxCellTemp', 'BCSThermistor1', 'BCSThermistor2']
    
    print(f"\n{'='*60}")
    print("FAULT CONDITION STATISTICS")
    print(f"{'='*60}")
    
    for feature in features:
        print(f"\n{feature}:")
        print(f"  PA6111 Faults - Mean: {faults1[feature].mean():.2f}, Std: {faults1[feature].std():.2f}")
        print(f"  PA7011 Faults - Mean: {faults2[feature].mean():.2f}, Std: {faults2[feature].std():.2f}")
        print(f"  Difference: {abs(faults1[feature].mean() - faults2[feature].mean()):.2f}")
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature Distribution During Faults: PA6111 vs PA7011', fontsize=16)
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        
        ax.hist(faults1[feature].dropna(), bins=30, alpha=0.5, label='PA6111', color='blue')
        ax.hist(faults2[feature].dropna(), bins=30, alpha=0.5, label='PA7011', color='red')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/fault_comparison.png')
    print("\n✓ Visualization saved to 'images/fault_comparison.png'")
    
    # Key insight
    print(f"\n{'='*60}")
    print("KEY INSIGHT")
    print(f"{'='*60}")
    
    temp_diff = abs(faults1['AMaxCellTemp'].mean() - faults2['AMaxCellTemp'].mean())
    if temp_diff > 5:
        print(f"⚠️  Large temperature difference detected ({temp_diff:.2f}°C)")
        print("The two units experience faults under different thermal conditions.")
        print("This explains why the single-dataset model failed to generalize.")
    else:
        print("Temperature patterns are similar. Issue may be in other features.")

if __name__ == "__main__":
    analyze_fault_differences()

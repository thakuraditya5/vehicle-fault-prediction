import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_and_evaluate():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv("vehicle_data.csv")
    
    # 2. EDA & Visualization
    print("Generating EDA plots...")
    
    # Correlation Matrix (only numeric columns)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.close()
    
    # Pairplot to see clusters
    sns.pairplot(df, hue="fault_type", palette="husl")
    plt.savefig("feature_pairplot.png")
    plt.close()

    # 3. Preprocessing
    X = df.drop("fault_type", axis=1)
    y = df["fault_type"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Model Training
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Plot
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # Feature Importance
    plt.figure(figsize=(8, 5))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    
    # 6. Advanced Visualizations
    import os
    if not os.path.exists("images"):
        os.makedirs("images")

    # 6a. Single Decision Tree Visualization
    print("Generating Decision Tree plot...")
    from sklearn.tree import plot_tree
    plt.figure(figsize=(20, 10))
    # Pick the first tree from the forest
    plot_tree(model.estimators_[0], 
              feature_names=X.columns,
              class_names=model.classes_,
              filled=True, 
              max_depth=3, 
              fontsize=10)
    plt.title("Single Decision Tree (Depth Limited)")
    plt.tight_layout()
    plt.savefig("images/decision_tree_structure.png")
    plt.close()

    # 6b. PCA Visualization (2D Projection)
    print("Generating PCA plot...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="bright", alpha=0.7)
    plt.title("Data Projection (PCA 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("images/pca_projection.png")
    plt.close()

    # Update save paths for existing plots to 'images/' folder
    # (Note: You'll need to manually move previous plot logic to use os.path.join("images", ...) 
    # or just let them stay in root. For cleanliness, let's move future saves to images/)

    # 7. Save Model
    joblib.dump(model, "vehicle_fault_model.pkl")
    print("\nModel saved to 'vehicle_fault_model.pkl'")
    print("Visualizations saved to 'images/' directory.")

if __name__ == "__main__":
    train_and_evaluate()

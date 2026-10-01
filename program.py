"""
==============================================================================
Program: program.py
Compiled from Jupyter Notebook: Untitled2.ipynb (C:/3rd semester/Python_Github/Untitled2.ipynb)
Description: Network Intrusion Detection Machine Learning Pipeline
- Dataset ingestion & preprocessing (handling NaNs, infinite values, encoding)
- Dimensionality reduction / Kernel approximation via Nystroem
- Model training & evaluation (CatBoost, DecisionTree, GradientBoosting, KNN, LDA)
- Comprehensive metrics evaluation across varying test sizes and kernels
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Classifiers
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Helper function to resolve dataset paths across common directories
def resolve_file(filename):
    possible_dirs = [
        ".",
        os.path.dirname(os.path.abspath(__file__)),
        r"C:\3rd semester\python_programm",
        r"C:\3rd semester\Python_Github"
    ]
    for d in possible_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return filename

# ==============================================================================
# 1. Dataset Loading
# ==============================================================================
file1 = resolve_file("Tuesday-WorkingHours.pcap_ISCX.csv")
file2 = resolve_file("Wednesday-workingHours.pcap_ISCX.csv")
file3 = resolve_file("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

print(f"Loading dataset 1 from: {file1}")
df1 = pd.read_csv(file1, low_memory=True)

print(f"Loading dataset 2 from: {file2}")
df2 = pd.read_csv(file2, low_memory=True)

print(f"Loading dataset 3 from: {file3}")
df3 = pd.read_csv(file3, low_memory=True)

# Concatenate dataframes
print("Concatenating datasets...")
dataset = pd.concat([df1, df2, df3], ignore_index=True)

# Clean column names (strip whitespace)
dataset.columns = dataset.columns.str.strip()

# Separate features (X) and target (y)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# ==============================================================================
# 2. Data Cleaning and Preprocessing
# ==============================================================================
print("Converting features to numeric...")
X = X.apply(pd.to_numeric, errors='coerce')

print("Replacing infinite values with NaN and imputing missing values...")
X.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

print("Encoding target labels...")
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# ==============================================================================
# 3. Model Configuration & Hyperparameters
# ==============================================================================
test_sizes = [0.2, 0.4, 0.6]
kernels = ['linear', 'sigmoid', 'rbf', 'cosine', 'poly']

algorithms = {
    'CatBoost': CatBoostClassifier(verbose=False),
    'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', random_state=0),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis()
}

# ==============================================================================
# 4. Training, Kernel Approximation (Nystroem) & Evaluation
# ==============================================================================
results = []

for test_size in test_sizes:
    print(f"\n{'='*60}\nRunning with Test Size: {test_size*100:.0f}%\n{'='*60}")

    # Split data once for the current test_size
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=0,
        stratify=y
    )

    for kernel in kernels:
        print(f"\n{'*'*50}\n  Applying Nystroem with Kernel: {kernel}\n{'*'*50}")

        # Apply Nystroem kernel approximation
        if kernel == 'sigmoid':
            k_pca = Nystroem(n_components=2, kernel=kernel, gamma=15, coef0=1, random_state=0)
        elif kernel == 'poly':
            k_pca = Nystroem(n_components=2, kernel=kernel, degree=3, coef0=1, random_state=0)
        else:
            k_pca = Nystroem(n_components=2, kernel=kernel, random_state=0)

        X_train_transformed = k_pca.fit_transform(X_train_split)
        X_test_transformed = k_pca.transform(X_test_split)

        for algo_name, classifier_model in algorithms.items():
            print(f"\n--- Running Algorithm: {algo_name} ---")

            if algo_name == 'KNeighborsClassifier':
                classifier = KNeighborsClassifier(n_neighbors=3)
            else:
                classifier = classifier_model

            try:
                classifier.fit(X_train_transformed, y_train_split)
                y_pred = classifier.predict(X_test_transformed)

                # Evaluate and print metrics
                cm = confusion_matrix(y_test_split, y_pred)
                accuracy = accuracy_score(y_test_split, y_pred)
                precision = precision_score(y_test_split, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_split, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_split, y_pred, average='weighted', zero_division=0)
                report = classification_report(y_test_split, y_pred, zero_division=0)

                print(f"Confusion Matrix for {algo_name} (Test Size: {test_size*100:.0f}%, Kernel: {kernel}):\n{cm}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"Classification Report:\n{report}")
                print("---------------------------------------------------")

                results.append({
                    'test_size': test_size,
                    'kernel': kernel,
                    'algorithm': algo_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report
                })

            except Exception as e:
                print(f"Error training {algo_name} with Test Size {test_size*100:.0f}% and Kernel {kernel}: {e}")
                results.append({
                    'test_size': test_size,
                    'kernel': kernel,
                    'algorithm': algo_name,
                    'error': str(e)
                })

# ==============================================================================
# 5. Display All Results
# ==============================================================================
print(f"\n{'='*60}\nDisplaying all computed results\n{'='*60}")

for result in results:
    algo_name = result.get('algorithm')
    test_size = result.get('test_size')
    kernel = result.get('kernel')

    if 'error' in result:
        print(f"\n--- Algorithm: {algo_name} (Test Size: {test_size*100:.0f}%, Kernel: {kernel}) ---")
        print(f"Error: {result['error']}")
    else:
        print(f"\n--- Algorithm: {algo_name} (Test Size: {test_size*100:.0f}%, Kernel: {kernel}) ---")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
        print(f"Classification Report:\n{result['classification_report']}")
        print("---------------------------------------------------")

# ==============================================================================
# 6. Specific Results for Test Size: 20%, Kernel: sigmoid
# ==============================================================================
print(f"\n{'='*60}\nSpecific Results for Test Size: 20%, Kernel: sigmoid\n{'='*60}")

for result in results:
    if result.get('test_size') == 0.2 and result.get('kernel') == 'sigmoid':
        algo_name = result.get('algorithm')
        if 'error' in result:
            print(f"\n--- Algorithm: {algo_name} ---")
            print(f"Error: {result['error']}")
        else:
            print(f"\n--- Algorithm: {algo_name} (Test Size: {result['test_size']*100:.0f}%, Kernel: {result['kernel']}) ---")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall: {result['recall']:.4f}")
            print(f"F1 Score: {result['f1_score']:.4f}")
            print(f"Classification Report:\n{result['classification_report']}")
            print("---------------------------------------------------")

# ==============================================================================
# 7. Summary DataFrame & Output
# ==============================================================================
results_df = pd.DataFrame(results)
print("\nFirst 5 rows of Results DataFrame:")
print(results_df.head())

# Save results to a CSV file alongside this program
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_results.csv")
results_df.to_csv(output_csv, index=False)
print(f"\nSaved results summary to: {output_csv}")

# ==============================================================================
# 8. Reference: Colab / GitHub Push Workflow (from notebook)
# ==============================================================================
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/GitHub_Projects
# !git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
# %cd /content/drive/MyDrive/GitHub_Projects/YOUR_REPOSITORY_NAME
# notebook_path = '/content/drive/MyDrive/Colab Notebooks/Untitled0.ipynb'
# !cp "{notebook_path}" .
# !git config --global user.email "you@example.com"
# !git config --global user.name "Your Name"
# !git add Untitled0.ipynb
# !git commit -m "Add initial Colab notebook"
# !git push origin main

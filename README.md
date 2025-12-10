# ML Project 2: Comparative Analysis of Classification Models

## Project Overview
This repository contains two distinct machine learning classification projects aimed at solving real-world prediction problems.
1.  **Forest Cover Type Prediction**: Classifying forest cover types based on strictly cartographic variables.
2.  **Smoker Status Prediction**: Predicting an individual's smoking status using a set of bio-signals.

Both projects involve comprehensive Exploratory Data Analysis (EDA), data preprocessing, and the implementation of multiple machine learning models to achieve high classification accuracy.

---

## Project 1: Forest Cover Type Prediction

### Goal
The objective is to predict the forest cover type (the predominant kind of tree cover) from cartographic variables only, without remotely sensed data. The dataset includes information on wilderness areas and soil types.

### Dataset
*   **File:** `Forest/covtype.csv`
*   **Target Variable:** `Cover_Type` (Integer classification 1-7)

### Methodology
1.  **Exploratory Data Analysis (EDA):** Analyzed distribution of cover types, relationships between elevation, slope, and soil types.
2.  **Preprocessing:** StandardScaler was used to normalize features.
3.  **Models Implemented:**
    *   **Logistic Regression:** Baseline model.
    *   **Support Vector Machine (Linear SVM):** For effective linear separation.
    *   **Neural Network (MLPClassifier):** A Multi-Layer Perceptron to capture non-linear complex relationships.

### Key Results
*   **MLP Neural Network** achieved the highest performance with an accuracy of approximately **90.6%**.
*   Logistic Regression and SVM provided baseline accuracies of ~72% and ~71% respectively.

---

## Project 2: Smoker Status Prediction

### Goal
To predict whether a person is a smoker or non-smoker based on various bio-signals such as height, weight, eyesight, hearing, and blood test results (cholesterol, hemoglobin, etc.).

### Dataset
*   **Training Data:** `Smoking/train_dataset.csv`
*   **Test Data:** `Smoking/test_dataset.csv`
*   **Target Variable:** `smoking` (Binary: 0 or 1)

### Methodology
1.  **Exploratory Data Analysis (EDA):** Investigated correlations between bio-signals and smoking status.
2.  **Preprocessing:** Handling missing values (if any), scaling numerical features using StandardScaler.
3.  **Models Implemented:**
    *   **Logistic Regression:** For binary classification baseline.
    *   **Support Vector Machine (SVM - RBF Kernel):** To handle non-linear decision boundaries.
    *   **Neural Network (MLPClassifier):** For deep pattern recognition.

### Key Results
*   **SVM (RBF Kernel)** demonstrated strong performance with ~75% accuracy on the test set.
*   **MLPClassifier** followed closely with ~74% accuracy.
*   **Logistic Regression** achieved ~72% accuracy.

---

## Installation & Usage

### Prerequisites
Ensure you have Python installed (version 3.8+ recommended). You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Notebooks
1.  Clone this repository.
2.  Navigate to the project directory:
    ```bash
    cd ML-Project-2
    ```
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4.  Open the desired notebook:
    *   For Forest Cover: `Forest/forest_cover_training.ipynb`
    *   For Smoking Status: `Smoking/smoking_training.ipynb`

### Directory Structure
```
ML-Project-2/
├── Forest/
│   ├── covtype.csv
│   ├── forest_cover_training.ipynb
│   └── ...
├── Smoking/
│   ├── train_dataset.csv
│   ├── test_dataset.csv
│   ├── smoking_training.ipynb
│   └── ...
└── README.md
```

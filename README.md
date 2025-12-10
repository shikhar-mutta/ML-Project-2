# Machine Learning Project: Forest Cover Type and Smoking Prediction

This repository contains two distinct machine learning projects:
1. **Forest Cover Type Prediction** - A multi-class classification task to predict forest cover types using cartographic variables
2. **Smoking Prediction** - A binary classification task to predict smoking status based on health examination data

## Project Structure

```
ML-Project-2/
│
├── Forest/
│   ├── covtype.csv            # Forest cover type dataset
│   ├── forest_cover_report_ready_eda.ipynb  # EDA for forest cover data
│   └── forest_cover_training.ipynb          # Model training and evaluation
│
└── Smoking/
    ├── train_dataset.csv      # Training data for smoking prediction
    ├── test_dataset.csv       # Test data for smoking prediction
    ├── full_updated_eda.ipynb # Comprehensive EDA for smoking data
    └── smoking_training.ipynb # Model development and evaluation
```

## 1. Forest Cover Type Prediction

### Overview
This project aims to predict forest cover types from cartographic variables. The dataset contains observations of forest cover types in the Roosevelt National Forest of northern Colorado, with the goal of predicting one of seven possible forest cover types.

### Dataset
- **Source**: [UCI Machine Learning Repository - Covertype](https://archive.ics.uci.edu/ml/datasets/covertype)
- **Size**: ~75MB (58,000+ instances)
- **Features**: 54 attributes including:
  - Elevation, aspect, slope
  - Horizontal and vertical distances to hydrology
  - Hillshade indices
  - Horizontal distance to roadways and fire points
  - Wilderness area designation (4 binary columns)
  - Soil type (40 binary columns)
- **Target**: Forest Cover Type (1-7)

### Models Implemented
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Multi-Layer Perceptron (MLP) Neural Network

### Key Files
- `forest_cover_report_ready_eda.ipynb`: Comprehensive exploratory data analysis
- `forest_cover_training.ipynb`: Model implementation and evaluation

## 2. Smoking Prediction

### Overview
This project focuses on predicting whether a person is a smoker based on various health examination metrics and personal information.

### Dataset
- **Training Set**: ~39,000 instances
- **Test Set**: ~9,800 instances
- **Features**: 23 attributes including:
  - Age, height, weight, waist circumference
  - Eyesight and hearing measurements
  - Blood pressure (systolic and diastolic)
  - Cholesterol and fasting blood sugar levels
  - HDL and LDL cholesterol
  - Hemoglobin and urine protein
  - Serum creatinine and AST/ALT levels
  - Gtp (gamma-glutamyl transpeptidase)
  - Dental caries
- **Target**: Smoking status (binary: smoker/non-smoker)

### Models Implemented
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Multi-Layer Perceptron (MLP) Neural Network

### Key Files
- `full_updated_eda.ipynb`: In-depth exploratory data analysis
- `smoking_training.ipynb`: Model implementation and evaluation

## Getting Started

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter

### Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd ML-Project-2
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run the notebooks in the respective project directories.

## Results

### Forest Cover Type Prediction
- Best performing model: [To be added after training]
- Accuracy: [To be added after training]
- Confusion matrix and classification report available in the notebook

### Smoking Prediction
- Best performing model: [To be added after training]
- Accuracy: [To be added after training]
- Key features and their importance in prediction

## Contributing

Feel free to submit issues and enhancement requests. Pull requests are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- UCI Machine Learning Repository for the Forest Cover Type dataset
- [Source of Smoking dataset if available]

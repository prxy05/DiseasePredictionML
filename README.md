# DiseasePredictionML

This is my Task 5 project where I worked on predicting diseases using machine learning models — specifically **Decision Tree** and **Random Forest**.  
The dataset contains patient health information and a target column that indicates the presence or absence of the disease.

## Project Overview

- **Goal**: To build classification models that can accurately predict disease presence based on given health features.
- **Models Used**:
  - Decision Tree Classifier
  - Random Forest Classifier
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Extra Work**:
  - Checked for overfitting by varying the depth of the decision tree.
  - Visualized the decision tree.
  - Analyzed feature importance using Random Forest.

## Dataset

- File: `DiseaseDataset.csv` (kept in the `data/` folder)
- Contains numerical and categorical health features such as:
- Age, cholesterol level, resting blood pressure, etc.
- Target column: `target` (0 = No disease, 1 = Disease)

## How To Run

**Install dependencies**
   ```bash
   pip install pandas numpy matplotlib scikit-learn

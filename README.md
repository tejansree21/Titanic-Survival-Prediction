# Titanic Survival Prediction using Machine Learning

This project predicts whether a passenger **survived or not** in the Titanic disaster using Machine Learning techniques.  
It uses the **built-in Titanic dataset** from Seaborn and applies a **Logistic Regression** model to classify survival outcomes.

---

## Dataset Overview

- **Source:** `seaborn.load_dataset('titanic')`
- **Total Samples:** ~891 passengers  
- **Target Variable:** `survived` (0 = Did not survive, 1 = Survived)

### Features Used
| Feature | Description |
|----------|-------------|
| `pclass` | Passenger class (1st, 2nd, 3rd) |
| `sex` | Gender of passenger |
| `age` | Age in years |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Ticket fare |
| `embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Project Workflow

1. **Load Dataset** using Seaborn’s built-in Titanic data  
2. **Data Cleaning**
   - Handle missing values (`age`, `embarked`)
   - Encode categorical variables (`sex`, `embarked`)
3. **Data Preprocessing**
   - Feature scaling using `StandardScaler`
4. **Model Training**
   - Train a **Logistic Regression** classifier
5. **Evaluation**
   - Accuracy, Confusion Matrix, and Classification Report
6. **Model Saving**
   - Save trained model and scaler using `joblib`

---

## Model Performance

| Metric | Score |
|:-------|:------:|
| **Accuracy** | ~0.79 |
| **Precision** | ~0.78 |
| **Recall** | ~0.72 |
| **F1-Score** | ~0.75 |

*(Scores may vary slightly each run.)*

---

## Visualization

Confusion Matrix of model predictions:

confusion_matrix.png

Example chart rendered in the notebook:
python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')

## How to run

1. Clone the repo
   git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction

2. Install dependencies
   pip install -r requirements.txt

3. Run the jupyter notebook
   jupyter notebook Titanic_Survival_Prediction.ipynb

Results
- Logistic Regression achieved nearly 79% accuracy
- Simple yet effective binary classification baseline
- Ready for deployment or comparison with advanced ML model


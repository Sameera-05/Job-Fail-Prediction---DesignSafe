# Job Failure Prediction using CatBoost & SHAP

This module provides a trained machine learning model to predict whether a job is likely to **FAIL or FINISH** before it is submitted to the HPC system. It uses historical job submission data and is powered by **CatBoost** for accuracy and **SHAP** for explainability.

---

## Repository Contents

- `job_failure_predictor.cbm` – Trained CatBoost classification model
- `model_metadata.pkl` – Contains `feature_names` and `cat_features` used for training
- `JobFailurePrediction.ipynb` – Full Jupyter notebook (EDA → preprocessing → modeling → SHAP)
- `job_failure_model_bundle.zip` – All necessary files zipped together for easy download

---

## How to Use the Model

### 1. Clone the repo or download the files

```bash
git clone https://github.com/your-org/job-failure-predictor.git
cd job-failure-predictor
```
Or manually download:
  -job_failure_predictor.cbm
  -model_metadata.pkl
### 2. Install Required Packages:
```bash
pip install catboost shap pandas scikit-learn joblib
```
### 3. Load Model and Predict:
``` python
from catboost import CatBoostClassifier
import pandas as pd
import joblib

# Load model
model = CatBoostClassifier()
model.load_model("job_failure_predictor.cbm")

# Load metadata
metadata = joblib.load("model_metadata.pkl")
cat_features = metadata["cat_features"]
feature_names = metadata["feature_names"]

# Sample job input (replace these with actual form inputs)
job_input = pd.DataFrame([{
    'system': 'Frontera',
    'app_id': 'LAMMPS',
    'app_version': '3Mar2020',
    'tenant': 'designsafe',
    '_tapisExecSystemId': 'frontera',
    'memoryMB': 64000,
    'nodeCount': 4,
    'processorsPerNode': 56,
    'maxRunTimeSeconds': 7200
}])

# Ensure all categorical columns are strings
for col in cat_features:
    job_input[col] = job_input[col].astype(str)

# Predict
prediction = model.predict(job_input)[0]
probability = model.predict_proba(job_input)[0][1]  # Probability of 'FAILED'

print("Prediction:", "FAILED" if prediction == 1 else "FINISHED")
print("Failure Probability Score:", round(probability, 3))
```

## How to Integrate into the Dashboard (Suggested Backend Flow)
- When user fills job submission form → extract relevant fields
- Construct the feature vector as shown above (job_input)
- Load model and predict() job outcome before submission
Display: "This job is likely to FAIL." with red warning if prediction is FAILED
Confidence score (e.g., 73% chance of failure)
(Optional) Add explanation using SHAP values for why it might fail

## Next Steps
 - Integrate model prediction into backend API (/predict_job_failure)
 - Connect API to job submission form
 - Display prediction warning in frontend
 - Optionally log user-submitted jobs + predictions for retraining

# Hospital Readmission Risk Predictor

This project predicts the probability of 30-day hospital readmission for diabetic patients using machine learning and a Streamlit interface.

## Files
- `app.py` - Streamlit app
- `config.py` - paths and constants
- `predict.py` - model loading and prediction
- `data_preprocessing.py` - data cleaning and preprocessing
- `train_models.py` - model training
- `evaluate_models.py` - threshold selection and evaluation
- `train_and_save.py` - end-to-end training script
- `models/` - saved model artifacts for deployment

## Run training
```bash
python train_and_save.py

# Saved Models Directory

This directory stores trained model files.

## Structure

Models are saved with timestamps and configuration information:
- `classical_model_YYYYMMDD_HHMMSS.pkl` - Trained classical models
- `quantum_model_YYYYMMDD_HHMMSS.pkl` - Trained quantum models

## Loading Models

Models can be loaded using pickle or joblib:

```python
import pickle

with open('saved_models/classical_model_20231201_120000.pkl', 'rb') as f:
    model = pickle.load(f)
```

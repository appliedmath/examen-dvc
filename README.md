# DVC and DagsHub exam

Prediction of the silica concentration in a mineral flotation process, with a DVC pipeline and DagsHub as remote storage.

## Structure

- `data/raw_data` raw data (`raw.csv`, tracked with DVC)
- `data/processed_data` train/test sets and scaled data
- `src/data` data scripts: `split_data.py`, `normalize_data.py`
- `src/models` model scripts: `grid_search.py`, `train_model.py`, `evaluate_model.py`
- `models` best parameters and trained model (`.pkl`)
- `metrics` evaluation metrics (`scores.json`)

## Pipeline

split -> normalize -> grid_search -> train -> evaluate

## Run

```bash
pip install -r requirements.txt
dvc pull
dvc repro
dvc metrics show
```

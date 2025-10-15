# The Scaffold-Free Blind Spot in Molecular Machine Learning: Toward Fairer Data Splitting

## Installation

To run the code, install the required Python packages with the following command:

```
bash
pip install matplotlib==3.9.4 \
            numpy==1.26.3 \
            pandas==2.2.3 \
            rdkit==2023.9.2 \
            scikit-learn==1.6.1 \
            tensorflow==2.15.0 \
            torch==2.7.1 \
            tqdm==4.67.1 \
            xgboost==2.1.4
```
            

## Data Processing Workflow

### 1. Extract Scaffold-Free Molecules (OOD)
Use `construct_scaffold_free_ood.py` to extract OOD scaffold-free molecules from the original dataset, generating `test.csv`; the remaining molecules are saved as `rest.csv`.

### 2. Scaffold-Based Split
Use `scaffold_kfold_split_rest.py` to perform a scaffold-based split on `rest.csv`, generating the training set `train.csv` and the validation set `valid.csv`.

### 3. Postprocessing and Hybrid Split
Use `postprocess_scaffold_split_hybrid.py` to set all scaffold-free molecules in `valid.csv` to `fold_1` for easy inspection, while performing the hybrid split.

## Model Training

### Three Models: `D-MPNN.py`, `FCNN.py`, and `XGBoost.py`
Note that `D-MPNN.py` must be run within the original repository of the authors (original repo: [https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)).


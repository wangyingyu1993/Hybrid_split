import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Global parameters
LEARNING_RATE = 0.05

# Dataset names and corresponding fold numbers
datasets_info = {
    "ESOL": 7,
    "FreeSolv": 5,
    "QM7": 10,
    "QM9": 10,
}

# Base directories for input and output
base_dir = Path("/media/aita4090/wyy/Hybrid_split/reordered_OOD_scaffold_split")
output_root = Path("/media/aita4090/wyy/Hybrid_split/scaffold_XGB_results")
output_root.mkdir(parents=True, exist_ok=True)

# ============ Feature Generation ============
def standardize(mol):
    """Standardize a molecule using RDKit MolStandardize tools."""
    try:
        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        te = rdMolStandardize.TautomerEnumerator()
        mol_final = te.Canonicalize(uncharged_parent_clean_mol)
    except:
        try:
            mol_final = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            mol_final = mol
    return mol_final


def smiles_to_features(smiles):
    """Convert a SMILES string to a numeric feature vector including
    molecular descriptors and FCFP4 fingerprints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = standardize(mol)
    features = []
    try:
        features.append(rdMolDescriptors.CalcTPSA(mol))
        features.append(rdMolDescriptors.CalcNumRings(mol))
        features.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
        features.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))
        features.append(rdMolDescriptors.CalcKappa1(mol))
        features.append(rdMolDescriptors.CalcKappa2(mol))
        features.append(rdMolDescriptors.CalcChi0n(mol))
        features.append(rdMolDescriptors.CalcChi1n(mol))
        features.append(rdMolDescriptors.CalcChi2n(mol))
        features.append(rdMolDescriptors.CalcChi3n(mol))
        features.append(rdMolDescriptors.CalcChi4n(mol))
        features.append(rdMolDescriptors.CalcLabuteASA(mol))
        features.extend(rdMolDescriptors.MQNs_(mol))
        features.extend(rdMolDescriptors.CalcAUTOCORR2D(mol))
    except:
        print("Failed to calculate RDKit descriptors")

    try:
        fcfp4_bit_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, useFeatures=True, nBits=1024
        )
        features.extend(list(fcfp4_bit_fp))
    except:
        print("Failed to calculate FCFP4 fingerprint")

    return np.array(features, dtype=float)


def extract_features(df, target_columns):
    """Extract features and targets from a DataFrame containing SMILES and target columns."""
    X, Y = [], []
    for _, row in df.iterrows():
        feats = smiles_to_features(row["molecules"])
        if feats is None:
            continue
        targets = [float(row[col]) for col in target_columns]
        X.append(feats)
        Y.append(targets)
    return np.array(X), np.array(Y)


# ============ Parameter Grid Search ============
param_spaces = {
    "ESOL": {"XGB": {'n_estimators': [100, 250, 500], 'max_depth': [3, 4, 5, 6]}},
    "FreeSolv": {"XGB": {'n_estimators': [100, 250, 500], 'max_depth': [3, 4, 5, 6]}},
    "QM7": {"XGB": {'n_estimators': [250, 500, 1000], 'max_depth': [3, 4, 5, 6, 7]}},
    "QM9": {"XGB": {'n_estimators': [500, 1000], 'max_depth': [6, 8]}},
}


def simple_grid_search(model_class, param_grid, X_train, y_train, X_val, y_val, **kwargs):
    """Simple grid search for best parameters based on validation MSE."""
    best_score, best_params = float("inf"), None
    from itertools import product
    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = model_class(**params, **kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = mean_squared_error(y_val, preds)
        if score < best_score:
            best_score, best_params = score, params
    return best_params


# ============ Main Training Loop ============
for dataset_name, kfolds in datasets_info.items():
    print(f"\n🚀 Processing dataset: {dataset_name}")
    dataset_dir = base_dir / dataset_name
    dataset_output = output_root / dataset_name
    dataset_output.mkdir(exist_ok=True, parents=True)

    for fold in range(1, kfolds + 1):
        print(f"\n🔁 Fold {fold}/{kfolds}")
        fold_dir = dataset_dir / f"fold_{fold}"
        train_df = pd.read_csv(fold_dir / "train.csv")
        valid_df = pd.read_csv(fold_dir / "valid.csv")
        test_df = pd.read_csv(fold_dir / "test.csv")

        target_columns = [col for col in train_df.columns if col != "molecules"]
        print(f"📌 Target columns: {target_columns}")

        X_train_all, Y_train_all = extract_features(train_df, target_columns)
        X_val_all, Y_val_all = extract_features(valid_df, target_columns)
        X_test_all, Y_test_all = extract_features(test_df, target_columns)

        # Cache best parameters for QM9 "gap" task
        best_params_cache = {}

        for task_idx, task in enumerate(target_columns):
            print(f"\n🎯 Training task: {task}")

            y_train = Y_train_all[:, task_idx]
            y_val = Y_val_all[:, task_idx]
            y_test = Y_test_all[:, task_idx]

            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
            y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

            # ========== XGBoost ==========
            if dataset_name == "QM9" and task != "gap" and "XGB" in best_params_cache:
                params = best_params_cache["XGB"]
            else:
                params = simple_grid_search(
                    XGBRegressor,
                    param_spaces[dataset_name]["XGB"],
                    X_train_all, y_train_s,
                    X_val_all, y_val_s,
                    learning_rate=LEARNING_RATE, n_jobs=6
                )
                if dataset_name == "QM9" and task == "gap":
                    best_params_cache["XGB"] = params

            # Train using xgb.train to track validation loss
            dtrain = xgb.DMatrix(X_train_all, label=y_train_s)
            dval = xgb.DMatrix(X_val_all, label=y_val_s)
            dtest = xgb.DMatrix(X_test_all)

            train_params = {
                "objective": "reg:squarederror",
                "eta": LEARNING_RATE,
                "nthread": 6,
                "max_depth": params["max_depth"],
            }
            num_boost_round = params["n_estimators"]

            evals_result = {}
            booster = xgb.train(
                train_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "validation")],
                evals_result=evals_result,
                verbose_eval=False
            )

            # Save validation loss curve
            val_rmse_curve = evals_result["validation"]["rmse"]
            val_loss_df = pd.DataFrame({
                "iteration": range(1, len(val_rmse_curve) + 1),
                "val_rmse": val_rmse_curve
            })
            val_loss_df.to_csv(dataset_output / f"fold_{fold}_val_loss_task_{task}.csv", index=False)

            # Predictions
            y_pred_s = booster.predict(dtest)
            y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Save predictions
            pred_df = pd.DataFrame({
                "molecules": test_df["molecules"],
                task: y_test,
                f"pred_{task}": y_pred
            })
            pred_df.to_csv(dataset_output / f"fold_{fold}_predictions_task_{task}.csv", index=False)

            # Save evaluation metrics
            metrics_df = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2}])
            metrics_df.to_csv(dataset_output / f"fold_{fold}_metrics_task_{task}.csv", index=False)

            print(f"✅ XGBoost Task {task} done — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

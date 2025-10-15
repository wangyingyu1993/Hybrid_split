import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Global parameters
EPOCHS = 300
BATCH_SIZE = 128

# Dataset names and corresponding fold numbers
datasets_info = {
    "ESOL": 7,
    "FreeSolv": 5,
    "QM7": 10,
    "QM9": 10,
}

# Base directories for input and output
base_dir = Path("/media/aita4090/wyy/Hybrid_split/reordered_OOD_scaffold_split")
output_root = Path("/media/aita4090/wyy/Hybrid_split/scaffold_FCNN_results")
output_root.mkdir(parents=True, exist_ok=True)

# ============ Molecular Standardization & Feature Generation ============
def standardize(mol):
    """Standardize a molecule using RDKit MolStandardize tools."""
    try:
        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger()
        uncharged = uncharger.uncharge(parent_clean_mol)
        te = rdMolStandardize.TautomerEnumerator()
        mol_final = te.Canonicalize(uncharged)
    except:
        try:
            mol_final = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            mol_final = mol
    return mol_final


def smiles_to_features(smiles):
    """Convert a SMILES string to a numeric feature vector including
    RDKit descriptors, Morgan fingerprints, and MACCS keys."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = standardize(mol)
    features = []

    # RDKit descriptors
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
        print("⚠️ Failed to calculate RDKit descriptors")

    # Morgan and MACCS fingerprints
    try:
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        features.extend(list(morgan_fp))
        features.extend(list(maccs_fp))
    except:
        print("⚠️ Failed to calculate Morgan/MACCS fingerprints")

    return np.array(features, dtype=float)


def extract_features(df, target_columns):
    """Extract features and target values from a DataFrame containing SMILES and target columns."""
    X, Y, valid_idx = [], [], []
    for i, row in df.iterrows():
        feats = smiles_to_features(row["molecules"])
        if feats is None:
            continue
        targets = [float(row[col]) for col in target_columns]
        X.append(feats)
        Y.append(targets)
        valid_idx.append(i)
    df = df.iloc[valid_idx].reset_index(drop=True)
    return np.array(X), np.array(Y), df


# ============ FCNN Model Definition ============
def build_model(input_dim, n_tasks):
    """Build a fully connected neural network model for regression."""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(30, activation='relu'),
        Dense(n_tasks)
    ])
    model.compile(optimizer=Adam(1e-3), loss='logcosh', metrics=['mse'])
    return model


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

        # Automatically identify target columns
        target_columns = [c for c in train_df.columns if c != "molecules"]
        print(f"📌 Target columns: {target_columns}")

        # Extract features
        X_train, Y_train, train_df = extract_features(train_df, target_columns)
        X_val, Y_val, valid_df = extract_features(valid_df, target_columns)
        X_test, Y_test, test_df = extract_features(test_df, target_columns)

        input_dim = X_train.shape[1]

        # === Train each task separately ===
        for task_idx, task in enumerate(target_columns):
            print(f"\n🎯 Training task: {task}")

            y_train = Y_train[:, task_idx].reshape(-1, 1)
            y_val = Y_val[:, task_idx].reshape(-1, 1)
            y_test = Y_test[:, task_idx].reshape(-1, 1)

            # Standardize target
            scaler_y = StandardScaler().fit(y_train)
            y_train_s = scaler_y.transform(y_train)
            y_val_s = scaler_y.transform(y_val)

            # Build model
            model = build_model(input_dim, n_tasks=1)
            checkpoint_path = dataset_output / f"best_fold_{fold}_task_{task}.h5"

            callbacks = [
                ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, mode='min'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, min_lr=1e-6),
                EarlyStopping(monitor='val_loss', patience=35, verbose=1, restore_best_weights=True)
            ]

            # Train model
            history = model.fit(
                X_train, y_train_s,
                validation_data=(X_val, y_val_s),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )

            # Test model
            model.load_weights(checkpoint_path)
            y_pred_s = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_s)

            # Save predictions
            result_df = pd.DataFrame({
                "molecules": test_df["molecules"],
                task: y_test.ravel(),
                f"pred_{task}": y_pred.ravel()
            })
            result_df.to_csv(dataset_output / f"fold_{fold}_predictions_task_{task}.csv", index=False)

            # Save loss curves
            metrics_df = pd.DataFrame({
                "epoch": range(1, len(history.history["loss"]) + 1),
                "train_loss": history.history["loss"],
                "val_loss": history.history["val_loss"]
            })
            metrics_df.to_csv(dataset_output / f"fold_{fold}_loss_task_{task}.csv", index=False)

            # Evaluation metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            metrics_summary = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2}])
            metrics_summary.to_csv(dataset_output / f"fold_{fold}_metrics_task_{task}.csv", index=False)

            print(f"✅ Fold {fold}, Task {task} done — MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

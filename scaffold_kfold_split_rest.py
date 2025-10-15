import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# === Path configuration ===
INPUT_ROOT = "/media/aita4090/wyy/Hybrid_split/dataset"
OUTPUT_ROOT = "/media/aita4090/wyy/Hybrid_split/OOD_scaffold_split"
SMILES_COLUMN = 'molecules'
MAX_SIZE = 60000000

np.random.seed(10000)


def get_split_ratio_and_kfolds(n):
    """
    Determine the train/validation split ratio and number of folds (k)
    based on the dataset size.

    Args:
        n (int): Number of samples in the dataset.

    Returns:
        tuple: ((train_ratio, valid_ratio), kfolds)
    """
    if n <= 1000:
        return (0.8, 0.2), 5  # train:valid ratio, number of folds
    elif n <= 3000:
        return (0.8, 0.2), 7
    elif n <= 14000:
        return (0.9, 0.1), 10
    else:
        return (0.9, 0.1), 10


def get_scaffold(smiles):
    """
    Generate the Bemis–Murcko scaffold for a given SMILES string.

    Args:
        smiles (str): SMILES representation of the molecule.

    Returns:
        str or None: SMILES of the scaffold, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


# === Main processing loop ===
# Iterate through dataset directories, each containing rest.csv and ood.csv
for dataset_name in os.listdir(INPUT_ROOT):
    dataset_dir = os.path.join(INPUT_ROOT, dataset_name)
    if not os.path.isdir(dataset_dir):
        continue

    rest_path = os.path.join(dataset_dir, "rest.csv")
    ood_path = os.path.join(dataset_dir, "ood.csv")

    # Skip if required files are missing
    if not os.path.exists(rest_path) or not os.path.exists(ood_path):
        print(f"Skipping {dataset_name}: missing rest.csv or ood.csv")
        continue

    print(f"Processing dataset: {dataset_name}")

    df_rest = pd.read_csv(rest_path).dropna(subset=[SMILES_COLUMN]).reset_index(drop=True)
    df_ood = pd.read_csv(ood_path).dropna(subset=[SMILES_COLUMN]).reset_index(drop=True)

    n = len(df_rest)
    (train_ratio, valid_ratio), KFOLDS = get_split_ratio_and_kfolds(n)
    print(f"rest.csv sample size: {n}, performing {KFOLDS}-fold cross-validation")

    # Generate scaffolds
    df_rest['scaffold'] = df_rest[SMILES_COLUMN].apply(get_scaffold)
    df_rest = df_rest.dropna(subset=['scaffold']).reset_index(drop=True)

    # Group by scaffold and shuffle group order
    scaffold_groups = (
        df_rest.groupby('scaffold')
        .apply(lambda x: x.sample(frac=1, random_state=1234).index.tolist())
        .tolist()
    )
    np.random.shuffle(scaffold_groups)

    # Distribute scaffold groups evenly across folds
    bucketed_groups = [[] for _ in range(KFOLDS)]
    for i, group in enumerate(scaffold_groups):
        bucketed_groups[i % KFOLDS].append(group)

    # Perform k-fold splitting: fold_i as validation, others as training
    for fold in range(KFOLDS):
        fold_dir = os.path.join(OUTPUT_ROOT, dataset_name, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        valid_indices = [idx for group in bucketed_groups[fold] for idx in group]
        train_indices = []
        for i in range(KFOLDS):
            if i != fold:
                for group in bucketed_groups[i]:
                    train_indices.extend(group)

        train_df = df_rest.loc[train_indices].sample(frac=1, random_state=10086).reset_index(drop=True)
        valid_df = df_rest.loc[valid_indices].sample(frac=1, random_state=10186).reset_index(drop=True)
        test_df = df_ood.sample(frac=1, random_state=10286).reset_index(drop=True)  # Fixed test set

        # Save CSVs (scaffold column removed)
        train_df.drop(columns=['scaffold']).to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        valid_df.drop(columns=['scaffold']).to_csv(os.path.join(fold_dir, "valid.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        print(f"[{dataset_name}] Fold {fold + 1} saved: "
              f"train({len(train_df)}), valid({len(valid_df)}), test({len(test_df)})")

    print(f"[{dataset_name}] {KFOLDS}-fold cross-validation completed\n")

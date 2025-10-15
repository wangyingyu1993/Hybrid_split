import os
import shutil
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

np.random.seed(1234)

# === CONFIGURATION ===
SMILES_COLUMN = 'molecules'
ROOT_IN = "/media/aita4090/wyy/Hybrid_split/OOD_scaffold_split"
ROOT_OUT_STEP1 = "/media/aita4090/wyy/Hybrid_split/reordered_OOD_scaffold_split"
ROOT_OUT_STEP2 = "/media/aita4090/wyy/Hybrid_split/reordered_OOD_hybrid_split"

# Clear previous output directories if they exist
if os.path.exists(ROOT_OUT_STEP1):
    shutil.rmtree(ROOT_OUT_STEP1)
if os.path.exists(ROOT_OUT_STEP2):
    shutil.rmtree(ROOT_OUT_STEP2)

# Dataset-specific train/valid split ratios
split_ratios = {
    'ESOL': (6, 1),
    'FreeSolv': (4, 1),
    'QM7': (9, 1),
    'QM9': (9, 1)
}

# === Scaffold generation function ===
def get_scaffold(smiles):
    """
    Generate a Bemis–Murcko scaffold from a SMILES string.
    Returns "None" if the molecule is invalid or has no scaffold.

    Args:
        smiles (str): Molecule SMILES string.
    Returns:
        str: Scaffold SMILES or "None".
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "None"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "None"
        return Chem.MolToSmiles(scaffold)
    except:
        return "None"

# === STEP 1: Move the fold containing scaffold-free molecules to fold_1 ===
print("=== Step 1: Moving fold with scaffold-free molecules to fold_1 ===")
for dataset in os.listdir(ROOT_IN):
    dataset_dir = os.path.join(ROOT_IN, dataset)
    if not os.path.isdir(dataset_dir):
        continue

    folds = [f for f in os.listdir(dataset_dir) if f.startswith("fold_")]
    if not folds:
        continue

    source_fold = None
    for fold in folds:
        fold_path = os.path.join(dataset_dir, fold)
        train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))
        valid_df = pd.read_csv(os.path.join(fold_path, "valid.csv"))

        train_df['scaffold'] = train_df[SMILES_COLUMN].apply(get_scaffold)
        valid_df['scaffold'] = valid_df[SMILES_COLUMN].apply(get_scaffold)

        train_none = train_df[train_df['scaffold'] == "None"]
        valid_none = valid_df[valid_df['scaffold'] == "None"]

        # If a fold contains scaffold-free molecules in validation but not in training, mark it for swapping
        if not train_none.empty:
            continue
        if not valid_none.empty:
            source_fold = fold
            break

    # If no swap is needed
    if source_fold is None or source_fold == "fold_1":
        shutil.copytree(dataset_dir, os.path.join(ROOT_OUT_STEP1, dataset), dirs_exist_ok=True)
        continue

    target_path = os.path.join(ROOT_OUT_STEP1, dataset)
    os.makedirs(target_path, exist_ok=True)

    # Swap fold_1 and source_fold
    original_fold1_path = os.path.join(dataset_dir, "fold_1")
    source_path = os.path.join(dataset_dir, source_fold)

    backup = os.path.join(target_path, "__temp__")
    shutil.copytree(original_fold1_path, backup)
    shutil.copytree(source_path, os.path.join(target_path, "fold_1"))
    shutil.copytree(backup, os.path.join(target_path, source_fold))
    shutil.rmtree(backup)

    # Copy remaining folds without modification
    for f in folds:
        if f not in ["fold_1", source_fold]:
            shutil.copytree(os.path.join(dataset_dir, f), os.path.join(target_path, f))

    print(f"[{dataset}] Swapped fold_1 <--> {source_fold}")

print("=== Step 1 Completed ===\n")

# === Step 1: Display fold data sizes ===
print("=== Step 1: Dataset Summary ===")
for dataset in os.listdir(ROOT_OUT_STEP1):
    dataset_dir = os.path.join(ROOT_OUT_STEP1, dataset)
    if not os.path.isdir(dataset_dir):
        continue

    folds = sorted([f for f in os.listdir(dataset_dir) if f.startswith("fold_")])
    for fold in folds:
        fold_path = os.path.join(dataset_dir, fold)
        train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))
        valid_df = pd.read_csv(os.path.join(fold_path, "valid.csv"))
        test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))
        print(f"[{dataset} - {fold}] train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}")

print("=== Step 1 Summary Completed ===\n")

# === STEP 2: Redistribute scaffold-free molecules and merge ===
print("=== Step 2: Redistributing scaffold-free molecules between train/valid ===")
for dataset in os.listdir(ROOT_OUT_STEP1):
    print(f"Processing dataset: {dataset}")
    dataset_dir = os.path.join(ROOT_OUT_STEP1, dataset)
    output_dir = os.path.join(ROOT_OUT_STEP2, dataset)
    os.makedirs(output_dir, exist_ok=True)

    if dataset not in split_ratios:
        print(f"No predefined split ratio for dataset: {dataset}")
        continue

    train_ratio, valid_ratio = split_ratios[dataset]
    total_parts = train_ratio + valid_ratio

    folds = sorted([f for f in os.listdir(dataset_dir) if f.startswith("fold_")])
    for fold in folds:
        fold_path = os.path.join(dataset_dir, fold)
        train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))
        valid_df = pd.read_csv(os.path.join(fold_path, "valid.csv"))
        test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))

        # Generate scaffolds
        train_df['scaffold'] = train_df[SMILES_COLUMN].apply(get_scaffold)
        valid_df['scaffold'] = valid_df[SMILES_COLUMN].apply(get_scaffold)

        # Separate scaffold-free and scaffold-containing molecules
        train_none = train_df[train_df['scaffold'] == "None"]
        valid_none = valid_df[valid_df['scaffold'] == "None"]
        total_none = pd.concat([train_none, valid_none], ignore_index=True).drop(columns=['scaffold'])

        keep_train = train_df[train_df['scaffold'] != "None"].drop(columns=['scaffold'])
        keep_valid = valid_df[valid_df['scaffold'] != "None"].drop(columns=['scaffold'])

        # Shuffle and re-split scaffold-free molecules according to ratio
        total_none = total_none.sample(frac=1, random_state=42).reset_index(drop=True)
        n_total = len(total_none)
        n_train = int(n_total * train_ratio / total_parts)

        new_train = total_none.iloc[:n_train]
        new_valid = total_none.iloc[n_train:]

        # Merge with existing scaffold-containing molecules
        final_train = pd.concat([keep_train, new_train], ignore_index=True).sample(frac=1, random_state=42)
        final_valid = pd.concat([keep_valid, new_valid], ignore_index=True).sample(frac=1, random_state=42)

        # Save results
        fold_out_path = os.path.join(output_dir, fold)
        os.makedirs(fold_out_path, exist_ok=True)
        final_train.to_csv(os.path.join(fold_out_path, "train.csv"), index=False)
        final_valid.to_csv(os.path.join(fold_out_path, "valid.csv"), index=False)
        test_df.to_csv(os.path.join(fold_out_path, "test.csv"), index=False)

        print(f"[{dataset} - {fold}] train: {len(final_train)}, valid: {len(final_valid)}, test: {len(test_df)}")

print("=== Step 2 Completed ===")

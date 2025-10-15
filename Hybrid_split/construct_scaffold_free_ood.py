import os
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

# ------------------- Configuration -------------------
input_dir = "/media/aita4090/wyy/Hybrid_split/dataset"
datasets = ["ESOL", "FreeSolv", "QM7", "QM9"]

# Create a Morgan fingerprint generator
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


# ------------------- Utility Functions -------------------
def get_scaffold(smiles):
    """Extract the Murcko scaffold from a SMILES string.
    Return 'None' if extraction fails or scaffold is empty."""
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


def smiles_to_fp(smiles):
    """Convert a SMILES string to a Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)


def max_similarity_to_set(fp, fps_set):
    """Compute the maximum Tanimoto similarity between a fingerprint
    and a set of fingerprints."""
    if fp is None or not fps_set:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_set)
    return max(sims) if sims else 0.0


# ------------------- Main Processing Loop -------------------
for dataset in datasets:
    file_path = os.path.join(input_dir, f"{dataset}.csv")
    print(f"Processing dataset: {dataset}")

    df = pd.read_csv(file_path)
    if "molecules" not in df.columns:
        raise ValueError(f"'molecules' column not found in {dataset}.csv")

    # Extract Murcko scaffolds
    df["scaffold"] = df["molecules"].apply(get_scaffold)

    # Identify molecules with missing scaffolds
    df_none = df[df["scaffold"] == "None"].copy()
    print(f"Number of molecules with scaffold=None: {len(df_none)}")

    # Compute fingerprints for all molecules
    fps_all = [smiles_to_fp(smi) for smi in df["molecules"]]
    fps_none = [smiles_to_fp(smi) for smi in df_none["molecules"]]

    # Select 10% of "None" molecules as OOD
    num_ood = max(1, int(len(df_none) * 0.1))
    ood_indices = []
    remaining_indices = set(range(len(df)))  # indices of the full dataset

    for idx_none, fp_none in zip(df_none.index, fps_none):
        if len(ood_indices) >= num_ood:
            break
        # Exclude the current molecule from similarity comparison
        candidate_indices = list(remaining_indices - {idx_none})
        candidate_fps = [fps_all[i] for i in candidate_indices if fps_all[i] is not None]

        sim_max = max_similarity_to_set(fp_none, candidate_fps)
        if sim_max <= 0.5:
            ood_indices.append(idx_none)
            remaining_indices.remove(idx_none)

    # Save OOD and rest sets
    ood_df = df.loc[ood_indices]
    rest_df = df.drop(index=ood_indices)

    # Remove the scaffold column
    if 'scaffold' in ood_df.columns:
        ood_df = ood_df.drop(columns=['scaffold'])
    if 'scaffold' in rest_df.columns:
        rest_df = rest_df.drop(columns=['scaffold'])

    output_dir = os.path.join(input_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)

    ood_df.to_csv(os.path.join(output_dir, "ood.csv"), index=False)
    rest_df.to_csv(os.path.join(output_dir, "rest.csv"), index=False)

    print(f"OOD set size: {len(ood_df)}, Rest set size: {len(rest_df)}\n")

print("All datasets processed successfully!")

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from chemprop import data, featurizers, models, nn

# -------------------------
# Custom Callback: logs metrics for each epoch
# -------------------------
class MetricsLogger(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        metrics_dict = {k: float(v) for k, v in logs.items() if isinstance(v, (float, torch.Tensor))}
        metrics_dict["epoch"] = trainer.current_epoch
        self.metrics.append(metrics_dict)

    def on_train_end(self, trainer, pl_module):
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.save_path, index=False)


# -------------------------
# Configuration
# -------------------------
datasets_info = {
    "QM7": 10,
    "ESOL": 7,
    "FreeSolv": 5,
    "QM9": 10
}

base_dir = Path("/media/aita4090/wyy/Hybrid_split/reordered_OOD_scaffold_split")
output_root = Path("/media/aita4090/wyy/Hybrid_split/scaffold_MPNN_results")
smiles_column = "molecules"
num_workers = 0

# -------------------------
# Main loop: iterate through datasets and folds
# -------------------------
for dataset_name, num_folds in datasets_info.items():
    dataset_dir = base_dir / dataset_name
    dataset_output_dir = output_root / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Processing dataset: {dataset_name} ({num_folds} folds)")

    for fold in range(1, num_folds + 1):
        print(f"\n🔁 Fold {fold}/{num_folds}")
        fold_dir = dataset_dir / f"fold_{fold}"

        df_train = pd.read_csv(fold_dir / "train.csv")
        df_val = pd.read_csv(fold_dir / "valid.csv")
        df_test = pd.read_csv(fold_dir / "test.csv")

        # Detect target columns
        target_columns = [col for col in df_train.columns if col != smiles_column]
        print(f"📌 Detected {len(target_columns)} tasks: {target_columns}")

        # Loop through each task (single-task training)
        for task in target_columns:
            print(f"\n🎯 Starting single-task training: {task}")

            # Construct datasets with single-column labels
            train_data = [data.MoleculeDatapoint.from_smi(smi, [df_train.loc[i, task]])
                          for i, smi in enumerate(df_train[smiles_column])]
            val_data = [data.MoleculeDatapoint.from_smi(smi, [df_val.loc[i, task]])
                        for i, smi in enumerate(df_val[smiles_column])]
            test_data = [data.MoleculeDatapoint.from_smi(smi, [df_test.loc[i, task]])
                         for i, smi in enumerate(df_test[smiles_column])]

            # Featurization and normalization
            featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
            train_dset = data.MoleculeDataset(train_data, featurizer)
            scaler = train_dset.normalize_targets()

            val_dset = data.MoleculeDataset(val_data, featurizer)
            val_dset.normalize_targets(scaler)

            test_dset = data.MoleculeDataset(test_data, featurizer)
            test_dset.normalize_targets(scaler)

            train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
            val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
            test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

            # Model construction (single-task)
            mp = nn.BondMessagePassing()
            agg = nn.MeanAggregation()
            output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

            ffn = nn.RegressionFFN(
                n_tasks=1,  # single-task
                output_transform=output_transform
            )

            metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
            model = models.MPNN(
                mp, agg, ffn,
                warmup_epochs=2,
                init_lr=1e-4,
                max_lr=1e-3,
                final_lr=1e-4,
                batch_norm=True,
                metrics=metric_list
            )

            # Checkpoint and metrics paths
            fold_task_dir = dataset_output_dir / f"checkpoints/fold_{fold}_task_{task}"
            fold_task_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(
                dirpath=fold_task_dir,
                filename="best-{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_last=True
            )

            metrics_logger = MetricsLogger(
                save_path=dataset_output_dir / f"fold_{fold}_metrics_task_{task}.csv"
            )
            early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

            trainer = pl.Trainer(
                logger=False,
                enable_progress_bar=True,
                enable_checkpointing=True,
                accelerator="auto",
                devices=1,
                max_epochs=20,
                callbacks=[checkpoint_callback, early_stopping, metrics_logger],
            )

            # Train model
            trainer.fit(model, train_loader, val_loader)

            # Test model
            trainer.test(model, dataloaders=test_loader)

            # Predict and save
            with torch.inference_mode():
                preds = trainer.predict(model, test_loader)
                preds = np.concatenate(preds, axis=0)

            df_pred = df_test.copy()
            df_pred[f"pred_{task}"] = preds[:, 0]  # single-task predictions have one column
            df_pred.to_csv(
                dataset_output_dir / f"fold_{fold}_predictions_task_{task}.csv", index=False
            )

            print(f"✅ Fold {fold}, single-task {task} completed and saved.")

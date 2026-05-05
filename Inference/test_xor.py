"""
XOR comparison FullIsingModule vs ModularNetwork on dimensions 1D-6D.
Single entry point: compare_xor_models_all_dimensions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from time import perf_counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import dotenv

dotenv.load_dotenv(dotenv_path=Path(__file__).parent / '.env')

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.utils.logger import Logger
from Inference.utils.plot import Plots
from Inference.utils.utils import (
    METRICS,
    build_annealing_settings,
    compute_metrics,
    flatten_logits,
    generate_xor_balanced,
    resolve_model_size,
    save_metrics_csv,
    standardize_train_test,
)
from full_ising_model.annealers import AnnealerType
from full_ising_model.full_ising_module import FullIsingModule
from NeuralNetwork.modular_network import ModularNetwork
from torch.utils.data import DataLoader, TensorDataset

logger = Logger()


def get_xor_params():
    """Load XOR experiment parameters strictly from .env (no defaults)."""
    return {
        'random_seed': int(os.environ['RANDOM_SEED']),
        'batch_size': int(os.environ['BATCH_SIZE']),
        'model_size': int(os.environ['MODEL_SIZE']),
        'minimum_model_size': int(os.environ['MINIMUM_MODEL_SIZE']),
        'num_ising_perceptrons': int(os.environ['NUM_ISING_PERCEPTRONS']),
        'epochs': int(os.environ['EPOCHS']),
        'lambda_init': float(os.environ['LAMBDA_INIT']),
        'offset_init': float(os.environ['OFFSET_INIT']),
        'hidden_nodes_offset_value': float(os.environ['HIDDEN_NODES_OFFSET_VALUE']),
        'lr_gamma': float(os.environ['LEARNING_RATE_GAMMA']),
        'lr_lambda': float(os.environ['LEARNING_RATE_LAMBDA']),
        'lr_offset': float(os.environ['LEARNING_RATE_OFFSET']),
        'lr_combiner': float(os.environ['LEARNING_RATE_COMBINER']),
        'sa_beta_range': [int(os.environ['SA_BETA_MIN']), int(os.environ['SA_BETA_MAX'])],
        'num_reads': int(os.environ['NUM_READS']),
        'sa_num_sweeps': int(os.environ['SA_NUM_SWEEPS']),
        'sa_sweeps_per_beta': int(os.environ['SA_SWEEPS_PER_BETA']),
        'num_workers': int(os.environ['NUM_THREADS']),
        'validation_interval': int(os.environ['VALIDATION_INTERVAL']),
        'n_samples_per_region': int(os.environ['N_SAMPLES_PER_REGION']),
        'test_size': float(os.environ['TEST_SIZE']),
        'partition_input': os.environ['PARTITION_INPUT'].lower() == 'true',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


def prepare_xor_data(dim, params):
    logger.info(f"Generating {dim}D XOR data...")
    X, y = generate_xor_balanced(
        dim=dim,
        n_samples_dim=params['n_samples_per_region'],
        shuffle=True,
        random_seed=params['random_seed'],
    )
    logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['test_size'],
        random_state=params['random_seed'],
        stratify=y,
    )
    logger.info(f"Train set: {len(y_train)} | Test set: {len(y_test)}")
    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, params):
    X_train, X_test = standardize_train_test(X_train, X_test)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=params['batch_size'],
        shuffle=True,
    )
    return X_tr, y_tr, X_te, y_te, train_loader


def _train_loop(model, optimizer, loss_fn, train_loader, X_te, y_test, params, label):
    training_losses = []
    validation_accuracies = []
    val_interval = params['validation_interval']

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(params['device'])
            y_batch = y_batch.to(params['device']).float()

            optimizer.zero_grad()
            pred = flatten_logits(model(x_batch))
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)

        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == params['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                logits = flatten_logits(model(X_te.to(params['device'])))
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                acc = accuracy_score(y_test, preds)
                validation_accuracies.append(acc)
            if (epoch + 1) % 10 == 0:
                logger.info(f"  [{label}] Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        else:
            if validation_accuracies:
                validation_accuracies.append(validation_accuracies[-1])

    model.eval()
    with torch.no_grad():
        logits = flatten_logits(model(X_te.to(params['device'])))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    final_accuracy = accuracy_score(y_test, preds)
    logger.info(f"  [{label}] Final Test Accuracy: {final_accuracy:.4f}")
    logger.info("\n" + classification_report(y_test, preds,
                                             target_names=['Class 0', 'Class 1'],
                                             zero_division=0))
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, preds)}")
    return final_accuracy, training_losses, validation_accuracies, preds, probs


def train_full_ising_model(dim, X_train, y_train, X_test, y_test, params,
                           annealer_type: AnnealerType = AnnealerType.SIMULATED):
    logger.info(f"\n{'='*60}\nTraining FullIsingModule on {dim}D XOR\n{'='*60}")
    model_size = resolve_model_size(X_train.shape[1], params)
    logger.info(f"Model size: {model_size}")

    _, _, X_te, _, train_loader = create_dataloaders(X_train, y_train, X_test, y_test, params)

    model = FullIsingModule(
        size_annealer=model_size,
        annealer_type=annealer_type,
        annealing_settings=build_annealing_settings(params),
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        num_workers=params['num_workers'],
        hidden_nodes_offset_value=params['hidden_nodes_offset_value'],
    ).to(params['device'])

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam([
        {'params': [model.gamma], 'lr': params['lr_gamma']},
        {'params': [model.lmd], 'lr': params['lr_lambda']},
        {'params': [model.offset], 'lr': params['lr_offset']},
    ])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    return _train_loop(model, optimizer, loss_fn, train_loader, X_te, y_test, params,
                       label=f'FullIsing/{dim}D')


def train_modular_network(dim, X_train, y_train, X_test, y_test, params,
                          annealer_type: AnnealerType = AnnealerType.SIMULATED):
    logger.info(f"\n{'='*60}\nTraining ModularNetwork on {dim}D XOR\n{'='*60}")
    model_size = resolve_model_size(X_train.shape[1], params)
    logger.info(f"Model size: {model_size}")

    _, _, X_te, _, train_loader = create_dataloaders(X_train, y_train, X_test, y_test, params)

    model = ModularNetwork(
        num_ising_perceptrons=params['num_ising_perceptrons'],
        size_annealer=model_size,
        annealing_settings=build_annealing_settings(params),
        annealer_type=annealer_type,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        hidden_nodes_offset_value=params['hidden_nodes_offset_value'],
        num_workers=params['num_workers'],
        combiner_bias=True,
        partition_input=params['partition_input'],
        random_seed=params['random_seed'],
    ).to(params['device'])

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer_groups = []
    for module in model.ising_perceptrons_layer:
        optimizer_groups.append({'params': [module.gamma], 'lr': params['lr_gamma']})
        optimizer_groups.append({'params': [module.lmd], 'lr': params['lr_lambda']})
        optimizer_groups.append({'params': [module.offset], 'lr': params['lr_offset']})
    optimizer_groups.append({'params': model.combiner_layer.parameters(),
                             'lr': params['lr_combiner']})
    optimizer = torch.optim.Adam(optimizer_groups)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    return _train_loop(model, optimizer, loss_fn, train_loader, X_te, y_test, params,
                       label=f'Modular/{dim}D')


def compare_xor_models_all_dimensions():
    """Compare FullIsingModule vs ModularNetwork on XOR dimensions 1D-6D."""
    params = get_xor_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = f'plots_xor_compare_{run_timestamp}'
    plotter = Plots(output_dir=out_dir)

    global logger
    logger = Logger(log_dir=out_dir)

    logger.info("\n" + "=" * 80)
    logger.info("XOR Comparison Suite - FullIsingModule vs ModularNetwork")
    logger.info("=" * 80)
    logger.info(f"Device: {params['device']}")
    logger.info(f"Batch size: {params['batch_size']} | Epochs: {params['epochs']} | Model size: {params['model_size']}")
    logger.info(f"Learning rates: gamma={params['lr_gamma']}, lambda={params['lr_lambda']}, "
                f"offset={params['lr_offset']}, combiner={params['lr_combiner']}")
    logger.info(f"Annealing: beta={params['sa_beta_range']}, reads={params['num_reads']}, "
                f"sweeps={params['sa_num_sweeps']}, workers={params['num_workers']}")

    dataset_names = []
    acc_single = []
    acc_modular = []
    metrics_single = []
    metrics_modular = []

    total_start = perf_counter()

    for dim in range(1, 7):
        ds_name = f"{dim}D"
        try:
            logger.info(f"\n[Dimension {ds_name}]")

            X_train, X_test, y_train, y_test = prepare_xor_data(dim, params)

            logger.info("  Training FullIsingModule...")
            _, _, _, _, probs_s = train_full_ising_model(
                dim, X_train, y_train, X_test, y_test, params
            )
            ms = compute_metrics(y_test, probs_s)
            logger.info(
                f"  FullIsingModule | acc={ms['accuracy']:.4f} prec={ms['precision']:.4f} "
                f"rec={ms['recall']:.4f} f1={ms['f1']:.4f} auc={ms['auc']:.4f}"
            )

            logger.info("  Training ModularNetwork...")
            _, _, _, _, probs_m = train_modular_network(
                dim, X_train, y_train, X_test, y_test, params
            )
            mm = compute_metrics(y_test, probs_m)
            logger.info(
                f"  ModularNetwork  | acc={mm['accuracy']:.4f} prec={mm['precision']:.4f} "
                f"rec={mm['recall']:.4f} f1={mm['f1']:.4f} auc={mm['auc']:.4f}"
            )

            dataset_names.append(ds_name)
            acc_single.append(ms['accuracy'])
            acc_modular.append(mm['accuracy'])
            metrics_single.append({m: [ms[m]] for m in METRICS})
            metrics_modular.append({m: [mm[m]] for m in METRICS})

            logger.info(
                f"  Diff(Modular-Single): {mm['accuracy'] - ms['accuracy']:+.4f}"
            )
        except Exception as e:
            logger.error(f"ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = perf_counter() - total_start
    logger.info(f"\n[COMPLETED in {total_time:.1f}s]")

    if not dataset_names:
        logger.error("No valid XOR dimensions processed.")
        return

    # --- Summary plots (reduced set) ---
    logger.info("Generating summary plots...")

    plotter.plot_tot_accuracy(acc_single, dataset_names)
    plotter.plot_compare_accuracy(
        acc_single, acc_modular, dataset_names,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )
    plotter.plot_parity_scatter(
        acc_single, acc_modular, dataset_names,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )

    # Per-metric arrays for bar comparison + heatmap
    m1 = {m: [metrics_single[i][m][0] for i in range(len(dataset_names))] for m in METRICS}
    m2 = {m: [metrics_modular[i][m][0] for i in range(len(dataset_names))] for m in METRICS}
    e1 = {m: [0.0 for _ in dataset_names] for m in METRICS}
    e2 = {m: [0.0 for _ in dataset_names] for m in METRICS}

    plotter.plot_metrics_bar_comparison(
        m1, m2, dataset_names, errors1=e1, errors2=e2,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )

    col_map = {'accuracy': 'Accuracy', 'precision': 'Precision',
               'recall': 'Recall', 'f1': 'F1', 'auc': 'AUC'}
    df_single = pd.DataFrame({col_map[m]: m1[m] for m in METRICS}, index=dataset_names)
    df_modular = pd.DataFrame({col_map[m]: m2[m] for m in METRICS}, index=dataset_names)
    plotter.plot_metrics_heatmap(
        df_single, df_modular,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )

    csv_path = save_metrics_csv(
        dataset_names, metrics_single, metrics_modular, plotter.output_dir,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )
    logger.info(f"Metrics CSV saved to: {csv_path}")
    logger.info(f"All plots saved to {plotter.output_dir}")


if __name__ == '__main__':
    compare_xor_models_all_dimensions()

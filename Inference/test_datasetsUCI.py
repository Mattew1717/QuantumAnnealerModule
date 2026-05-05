"""
K-Fold cross-validation comparison FullIsingModule vs ModularNetwork on UCI datasets.
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon
import dotenv

dotenv.load_dotenv(dotenv_path=Path(__file__).parent / '.env')

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.utils.logger import Logger
from Inference.utils.dataset_manager import DatasetManager
from Inference.utils.plot import Plots
from Inference.utils.utils import (
    METRICS,
    build_annealing_settings,
    compute_metrics,
    flatten_logits,
    resolve_model_size,
    save_metrics_csv,
    set_global_seed,
    standardize_train_test,
)
from full_ising_model.annealers import AnnealerType
from full_ising_model.full_ising_module import FullIsingModule
from NeuralNetwork.modular_network import ModularNetwork

logger = Logger()


def get_env_params():
    """Strict load of all parameters from .env."""
    return {
        'random_seed': int(os.environ['RANDOM_SEED']),
        'batch_size': int(os.environ['BATCH_SIZE']),
        'model_size': int(os.environ['MODEL_SIZE']),
        'minimum_model_size': int(os.environ['MINIMUM_MODEL_SIZE']),
        'partition_input': os.environ['PARTITION_INPUT'].lower() == 'true',
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
        'k_folds': int(os.environ['K_FOLDS']),
        'num_workers': int(os.environ['NUM_THREADS']),
        'print_interval': int(os.environ['PRINT_INTERVAL']),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


def print_config(params, run_timestamp):
    logger.info(f"\n[RUN {run_timestamp}]")
    logger.info(f"Device: {params['device']} | Batch: {params['batch_size']} | "
                f"Epochs: {params['epochs']} | K-Folds: {params['k_folds']}")
    model_size_label = 'auto' if params['model_size'] == -1 else params['model_size']
    logger.info(f"Model: size={model_size_label}, perceptrons={params['num_ising_perceptrons']}, "
                f"partition={params['partition_input']}, workers={params['num_workers']}")
    logger.info(f"LR: gamma={params['lr_gamma']:.4f}, lambda={params['lr_lambda']:.4f}, "
                f"offset={params['lr_offset']:.4f}, combiner={params['lr_combiner']:.4f}")
    logger.info(f"SA: beta={params['sa_beta_range']}, reads={params['num_reads']}, "
                f"sweeps={params['sa_num_sweeps']}")
    logger.info(f"Init: lambda={params['lambda_init']:.4f}, offset={params['offset_init']:.4f}\n")


def _train_eval(model, optimizer, loss_fn, train_loader, X_test_t, y_test_arr, params):
    training_losses = []
    print_interval = params['print_interval']

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
        if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == params['epochs'] - 1:
            logger.info(f"  Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = flatten_logits(model(X_test_t.to(params['device'])))
        final_probs = torch.sigmoid(logits).cpu().numpy()
        final_preds = (final_probs >= 0.5).astype(int)
        final_acc = accuracy_score(y_test_arr, final_preds)

    return final_acc, final_probs, training_losses


def train_full_ising_module(X_train, y_train, X_test, y_test, params):
    X_train, X_test = standardize_train_test(X_train, X_test)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=params['batch_size'],
        shuffle=True,
    )

    size = resolve_model_size(X_train.shape[1], params)
    logger.info(f"Using model size: {size}")

    model = FullIsingModule(
        size_annealer=size,
        annealer_type=AnnealerType.SIMULATED,
        annealing_settings=build_annealing_settings(params),
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        num_workers=params['num_workers'],
        hidden_nodes_offset_value=params['hidden_nodes_offset_value'],
    ).to(params['device'])

    optimizer = torch.optim.Adam([
        {'params': [model.gamma], 'lr': params['lr_gamma']},
        {'params': [model.lmd], 'lr': params['lr_lambda']},
        {'params': [model.offset], 'lr': params['lr_offset']},
    ])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    return _train_eval(model, optimizer, loss_fn, train_loader, X_test_t, y_test, params)


def train_modular_network(X_train, y_train, X_test, y_test, params):
    X_train, X_test = standardize_train_test(X_train, X_test)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=params['batch_size'],
        shuffle=True,
    )

    size = resolve_model_size(X_train.shape[1], params)
    logger.info(f"Using model size: {size}")

    model = ModularNetwork(
        num_ising_perceptrons=params['num_ising_perceptrons'],
        size_annealer=size,
        annealing_settings=build_annealing_settings(params),
        annealer_type=AnnealerType.SIMULATED,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        hidden_nodes_offset_value=params['hidden_nodes_offset_value'],
        num_workers=params['num_workers'],
        combiner_bias=True,
        partition_input=params['partition_input'],
        random_seed=params['random_seed'],
    ).to(params['device'])

    optimizer_groups = []
    for module in model.ising_perceptrons_layer:
        optimizer_groups.append({'params': [module.gamma], 'lr': params['lr_gamma']})
        optimizer_groups.append({'params': [module.lmd], 'lr': params['lr_lambda']})
        optimizer_groups.append({'params': [module.offset], 'lr': params['lr_offset']})
    optimizer_groups.append({'params': model.combiner_layer.parameters(),
                             'lr': params['lr_combiner']})
    optimizer = torch.optim.Adam(optimizer_groups)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    return _train_eval(model, optimizer, loss_fn, train_loader, X_test_t, y_test, params)


def compare_models():
    """Compare FullIsingModule vs ModularNetwork on all UCI datasets via K-Fold CV."""
    params = get_env_params()
    set_global_seed(params['random_seed'])
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = f'plots_{run_timestamp}'
    plotter = Plots(output_dir=out_dir)

    global logger
    logger = Logger(log_dir=out_dir)
    print_config(params, run_timestamp)

    dataset_manager = DatasetManager()
    datasets_dir = os.path.join(os.path.dirname(__file__), 'Datasets')
    csv_files = sorted(glob.glob(os.path.join(datasets_dir, '*.csv')))
    if not csv_files:
        logger.error(f"No CSV files found in {datasets_dir}")
        return

    dataset_names = []
    single_accuracies_all = []
    modular_accuracies_all = []
    single_losses_all = []
    modular_losses_all = []
    all_single_metrics = []
    all_modular_metrics = []

    total_start = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info(f"\n[Dataset {idx}/{len(csv_files)}: {name}]")

        try:
            X, y = dataset_manager.load_csv_dataset(csv_path, random_seed=params['random_seed'])
            folds = dataset_manager.generate_k_folds(X, y, params['k_folds'],
                                                    random_seed=params['random_seed'])
            logger.info(f"Running {params['k_folds']}-Fold Cross Validation")

            single_accs = []
            modular_accs = []
            single_losses = []
            modular_losses = []
            single_fold_metrics = {m: [] for m in METRICS}
            modular_fold_metrics = {m: [] for m in METRICS}

            for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds, 1):
                logger.info(f"Fold {fold_idx}/{params['k_folds']}")

                logger.info("  Training FullIsingModule...")
                _, probs_s, loss_s = train_full_ising_module(
                    X_train, y_train, X_test, y_test, params
                )
                ms = compute_metrics(y_test, probs_s)
                logger.info(f"  Single  | acc={ms['accuracy']:.4f} prec={ms['precision']:.4f} "
                            f"rec={ms['recall']:.4f} f1={ms['f1']:.4f} auc={ms['auc']:.4f}")

                logger.info("  Training ModularNetwork...")
                _, probs_m, loss_m = train_modular_network(
                    X_train, y_train, X_test, y_test, params
                )
                mm = compute_metrics(y_test, probs_m)
                logger.info(f"  Modular | acc={mm['accuracy']:.4f} prec={mm['precision']:.4f} "
                            f"rec={mm['recall']:.4f} f1={mm['f1']:.4f} auc={mm['auc']:.4f}")

                for k in METRICS:
                    single_fold_metrics[k].append(ms[k])
                    modular_fold_metrics[k].append(mm[k])

                single_accs.append(ms['accuracy'])
                modular_accs.append(mm['accuracy'])
                single_losses.append(loss_s)
                modular_losses.append(loss_m)

            dataset_names.append(name)
            single_accuracies_all.append(single_accs)
            modular_accuracies_all.append(modular_accs)
            single_losses_all.append(single_losses)
            modular_losses_all.append(modular_losses)
            all_single_metrics.append(single_fold_metrics)
            all_modular_metrics.append(modular_fold_metrics)

            mean_acc_s = np.mean(single_accs)
            mean_acc_m = np.mean(modular_accs)
            std_acc_s = np.std(single_accs)
            std_acc_m = np.std(modular_accs)
            logger.info(f"\nResults | Single: {mean_acc_s:.4f}±{std_acc_s:.4f} | "
                        f"Modular: {mean_acc_m:.4f}±{std_acc_m:.4f} | "
                        f"Diff: {mean_acc_m - mean_acc_s:+.4f}\n")

            plotter.plot_loss(single_losses[0], f"{name}_single")
            plotter.plot_loss(modular_losses[0], f"{name}_modular")

        except Exception as e:
            logger.error(f"ERROR: {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start
    logger.info(f"\n[COMPLETED in {total_time:.1f}s]\n")

    if not dataset_names:
        logger.error("No valid datasets processed.")
        return

    logger.info("Generating summary plots...")

    single_means = [np.mean(accs) for accs in single_accuracies_all]
    modular_means = [np.mean(accs) for accs in modular_accuracies_all]

    plotter.plot_tot_accuracy(single_means, dataset_names)
    plotter.plot_compare_accuracy(
        single_means, modular_means, dataset_names,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )
    plotter.plot_parity_scatter(
        single_means, modular_means, dataset_names,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )

    try:
        stat, p_value = wilcoxon(single_means, modular_means)
        logger.info(f"\nWilcoxon signed-rank test: statistic={stat:.4f}, p-value={p_value:.4f}")
    except Exception as e:
        logger.error(f"Wilcoxon test error: {e}")

    plotter.box_plot(
        single_accuracies_all, dataset_names,
        title='FullIsingModule: K-Fold accuracy distribution',
        filename='box_plot_single_model',
    )
    plotter.box_plot(
        modular_accuracies_all, dataset_names,
        title='ModularNetwork: K-Fold accuracy distribution',
        filename='box_plot_modular_network',
    )

    csv_path = save_metrics_csv(
        dataset_names, all_single_metrics, all_modular_metrics, plotter.output_dir,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )
    logger.info(f"Metrics CSV saved to: {csv_path}")

    per_fold_rows = []
    for ds_idx, ds in enumerate(dataset_names):
        for fold_idx in range(params['k_folds']):
            per_fold_rows.append({
                'dataset': ds, 'model': 'FullIsingModule', 'fold': fold_idx + 1,
                **{m: all_single_metrics[ds_idx][m][fold_idx] for m in METRICS},
            })
            per_fold_rows.append({
                'dataset': ds, 'model': 'ModularNetwork', 'fold': fold_idx + 1,
                **{m: all_modular_metrics[ds_idx][m][fold_idx] for m in METRICS},
            })
    per_fold_path = os.path.join(str(plotter.output_dir), 'metrics_per_fold.csv')
    pd.DataFrame(per_fold_rows).to_csv(per_fold_path, index=False, float_format='%.4f')
    logger.info(f"Per-fold metrics CSV saved to: {per_fold_path}")

    m1 = {m: [np.nanmean(all_single_metrics[i][m]) for i in range(len(dataset_names))]
          for m in METRICS}
    m2 = {m: [np.nanmean(all_modular_metrics[i][m]) for i in range(len(dataset_names))]
          for m in METRICS}
    e1 = {m: [np.nanstd(all_single_metrics[i][m]) for i in range(len(dataset_names))]
          for m in METRICS}
    e2 = {m: [np.nanstd(all_modular_metrics[i][m]) for i in range(len(dataset_names))]
          for m in METRICS}

    col_map = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
               'f1': 'F1', 'auc': 'AUC'}
    df_single = pd.DataFrame({col_map[m]: m1[m] for m in METRICS}, index=dataset_names)
    df_modular = pd.DataFrame({col_map[m]: m2[m] for m in METRICS}, index=dataset_names)

    plotter.plot_metrics_bar_comparison(
        m1, m2, dataset_names, errors1=e1, errors2=e2,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )
    plotter.plot_metrics_heatmap(
        df_single, df_modular,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )
    plotter.plot_combined_boxplot(
        [all_single_metrics[i]['accuracy'] for i in range(len(dataset_names))],
        [all_modular_metrics[i]['accuracy'] for i in range(len(dataset_names))],
        dataset_names,
        model_name1='FullIsingModule', model_name2='ModularNetwork',
    )

    logger.info(f"All plots saved to {plotter.output_dir}")


if __name__ == '__main__':
    compare_models()

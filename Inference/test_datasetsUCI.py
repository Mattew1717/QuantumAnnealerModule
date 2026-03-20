import os
import sys
import glob
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
import dotenv
from scipy.stats import wilcoxon, ttest_rel

# Load environment variables
dotenv.load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Add repository root to sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.utils.logger import Logger
from Inference.utils.dataset_manager import DatasetManager
from Inference.utils.plot import Plot
from Inference.utils.utils import flatten_logits, compute_metrics, save_metrics_csv, METRICS
from full_ising_model.full_ising_module import FullIsingModule
from ModularNetwork.Network_1L import MultiIsingNetwork
from full_ising_model.annealers import AnnealingSettings, AnnealerType

logger = Logger()


def get_env_params():
    """Load all parameters from .env file."""
    return {
        'random_seed': int(os.getenv('RANDOM_SEED')),
        'batch_size': int(os.getenv('BATCH_SIZE')),
        'model_size': int(os.getenv('MODEL_SIZE')),
        'minimum_model_size': int(os.getenv('MINIMUM_MODEL_SIZE')),
        'partition_input': os.getenv('PARTITION_INPUT').lower() == 'true',
        'num_ising_perceptrons': int(os.getenv('NUM_ISING_PERCEPTRONS')),
        'epochs': int(os.getenv('EPOCHS')),
        'lambda_init': float(os.getenv('LAMBDA_INIT')),
        'offset_init': float(os.getenv('OFFSET_INIT')),
        'lr_gamma': float(os.getenv('LEARNING_RATE_GAMMA')),
        'lr_lambda': float(os.getenv('LEARNING_RATE_LAMBDA')),
        'lr_offset': float(os.getenv('LEARNING_RATE_OFFSET')),
        'lr_combiner': float(os.getenv('LEARNING_RATE_COMBINER')),
        'sa_beta_range': [int(os.getenv('SA_BETA_MIN')), int(os.getenv('SA_BETA_MAX'))],
        'sa_num_reads': int(os.getenv('NUM_READS')),
        'sa_num_sweeps': int(os.getenv('SA_NUM_SWEEPS')),
        'sa_sweeps_per_beta': int(os.getenv('SA_SWEEPS_PER_BETA')),
        'k_folds': int(os.getenv('K_FOLDS')),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }


def print_config(params, run_timestamp):
    """Print configuration parameters."""
    logger.info(f"\n[RUN {run_timestamp}]")
    logger.info(f"Device: {params['device']} | Batch: {params['batch_size']} | Epochs: {params['epochs']} | K-Folds: {params['k_folds']}")
    logger.info(f"Model: size={"auto" if params['model_size'] == -1 else params['model_size']}, perceptrons={params['num_ising_perceptrons']}, partition={params['partition_input']}")
    logger.info(f"LR: gamma={params['lr_gamma']:.4f}, lambda={params['lr_lambda']:.4f}, offset={params['lr_offset']:.4f}, combiner={params['lr_combiner']:.4f}")
    logger.info(f"SA: beta={params['sa_beta_range']}, reads={params['sa_num_reads']}, sweeps={params['sa_num_sweeps']}")
    logger.info(f"Init: lambda={params['lambda_init']:.4f}, offset={params['offset_init']:.4f}\n")


def train_single_model(X_train, y_train, X_test, y_test, params):
    """Train and evaluate single Ising model."""

    # Standardization without leakage
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Create datasets without manual resize (FullIsingModule handles it)
    from Inference.utils.dataset_manager import SimpleDataset
    train_set = SimpleDataset()
    train_set.x = torch.tensor(X_train, dtype=torch.float32)
    train_set.y = torch.tensor(y_train, dtype=torch.float32)
    train_set.data_size = X_train.shape[1]
    train_set.len = len(y_train)

    test_set = SimpleDataset()
    test_set.x = torch.tensor(X_test, dtype=torch.float32)
    test_set.y = torch.tensor(y_test, dtype=torch.float32)
    test_set.data_size = X_test.shape[1]
    test_set.len = len(y_test)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_set.x, train_set.y),
        batch_size=params['batch_size'],
        shuffle=True
    )

    # Setup simulated annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['sa_num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.num_sweeps_per_beta = params['sa_sweeps_per_beta']

    if params['model_size'] == -1:
        size = X_train.shape[1] if X_train.shape[1] > params['minimum_model_size'] else params['minimum_model_size']
    else:
        size = params['model_size']
    logger.info(f"Using model size: {size}")
    
    # Create model
    model = FullIsingModule(
        size_annealer=size,
        annealer_type=AnnealerType.SIMULATED,
        annealing_settings=SA_settings,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init']
    ).to(params['device'])

    # Setup optimizer
    optimizer = torch.optim.Adam([
        {'params': [model.gamma], 'lr': params['lr_gamma']},
        {'params': [model.lmd], 'lr': params['lr_lambda']},
        {'params': [model.offset], 'lr': params['lr_offset']},
    ])

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    training_losses = []
    validation_accuracies = []
    start_time = time.time()

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
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
        if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f}")
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            preds_tensor = flatten_logits(model(test_set.x.to(params['device'])))
            probs = torch.sigmoid(preds_tensor).cpu().numpy()
            predictions = np.where(probs < 0.5, 0, 1)
            epoch_accuracy = accuracy_score(y_test, predictions)
            validation_accuracies.append(epoch_accuracy)

    training_time = time.time() - start_time

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        preds_tensor = flatten_logits(model(test_set.x.to(params['device'])))
        final_probs = torch.sigmoid(preds_tensor).cpu().numpy()

    final_accuracy = validation_accuracies[-1]

    return final_accuracy, final_probs, training_losses, validation_accuracies


def train_neural_net(X_train, y_train, X_test, y_test, params):
    """Train and evaluate neural Ising network."""

    # Standardization without leakage
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Create datasets without manual resize (FullIsingModule handles it)
    from Inference.utils.dataset_manager import SimpleDataset
    train_set = SimpleDataset()
    train_set.x = torch.tensor(X_train, dtype=torch.float32)
    train_set.y = torch.tensor(y_train, dtype=torch.float32)
    train_set.data_size = X_train.shape[1]
    train_set.len = len(y_train)

    test_set = SimpleDataset()
    test_set.x = torch.tensor(X_test, dtype=torch.float32)
    test_set.y = torch.tensor(y_test, dtype=torch.float32)
    test_set.data_size = X_test.shape[1]
    test_set.len = len(y_test)

    # Determine model size
    if params['model_size'] == -1:
        size = X_train.shape[1] if X_train.shape[1] > params['minimum_model_size'] else params['minimum_model_size']
    else:
        size = params['model_size']
    logger.info(f"Using model size: {size}")

    # Handle partitioning: if partition_input, we need to manually resize for partitioning
    # Otherwise, FullIsingModule will handle the resize automatically
    if params['partition_input']:
        target_size = size * params['num_ising_perceptrons']

        # Resize for partitioning
        from Inference.utils.dataset_manager import HiddenNodesInitialization
        hn = HiddenNodesInitialization('function')
        hn.function = SimpleDataset.offset
        hn.fun_args = [-0.02]

        train_set.resize(target_size, hn)
        test_set.resize(target_size, hn)
        train_set.data_size = target_size
        test_set.data_size = target_size

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_set.x, train_set.y),
        batch_size=params['batch_size'],
        shuffle=True
    )

    # Setup simulated annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['sa_num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.num_sweeps_per_beta = params['sa_sweeps_per_beta']
    # Create model
    model = MultiIsingNetwork(
        num_ising_perceptrons=params['num_ising_perceptrons'],
        size_annealer=size,
        annealing_settings=SA_settings,
        annealer_type=AnnealerType.SIMULATED,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        partition_input=params['partition_input'],
    ).to(params['device'])

    # Setup optimizer
    optimizer_grouped_parameters = []
    for p_idx, single_module in enumerate(model.ising_perceptrons_layer):
        optimizer_grouped_parameters.append({
            'params': [single_module.gamma],
            'lr': params['lr_gamma']
        })
        optimizer_grouped_parameters.append({
            'params': [single_module.lmd],
            'lr': params['lr_lambda']
        })
        optimizer_grouped_parameters.append({
            'params': [single_module.offset],
            'lr': params['lr_offset']
        })
    optimizer_grouped_parameters.append({
        'params': model.combiner_layer.parameters(),
        'lr': params['lr_combiner']
    })

    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    training_losses = []
    validation_accuracies = []
    start_time = time.time()

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(params['device'])
            y_batch = y_batch.to(params['device']).float()

            optimizer.zero_grad()
            pred = flatten_logits(model(x_batch))
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f}")
        training_losses.append(avg_loss)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            preds_tensor = flatten_logits(model(test_set.x.to(params['device'])))
            probs = torch.sigmoid(preds_tensor).cpu().numpy()
            predictions = np.where(probs < 0.5, 0, 1)
            epoch_accuracy = accuracy_score(y_test, predictions)
            validation_accuracies.append(epoch_accuracy)

    training_time = time.time() - start_time

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        preds_tensor = flatten_logits(model(test_set.x.to(params['device'])))
        final_probs = torch.sigmoid(preds_tensor).cpu().numpy()

    final_accuracy = validation_accuracies[-1]

    return final_accuracy, final_probs, training_losses, validation_accuracies


def paired_ttest_kfold(single_fold_metrics, neural_fold_metrics, dataset_name):
    """
    Paired t-test (ttest_rel) between FullIsingModel and IsingNet across the same K folds.

    Args:
        single_fold_metrics: dict {metric: [fold_0_val, ..., fold_k_val]} for FullIsingModule
        neural_fold_metrics:  dict {metric: [fold_0_val, ..., fold_k_val]} for MultiIsingNetwork
        dataset_name:         str, used for logging

    Returns:
        results: dict {metric: {'statistic': float, 'p_value': float, 'significant': bool}}
    """
    results = {}
    logger.info(f"\n[Paired t-test] Dataset: {dataset_name}")
    logger.info(f"  {'Metric':<12} {'t-stat':>10} {'p-value':>10} {'Significant':>12}")
    logger.info(f"  {'-'*46}")

    for metric in METRICS:
        a = np.array(single_fold_metrics[metric], dtype=float)
        b = np.array(neural_fold_metrics[metric], dtype=float)

        # Skip if any NaN or fewer than 2 folds
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() < 2:
            logger.info(f"  {metric:<12} {'N/A':>10} {'N/A':>10} {'insufficient data':>12}")
            results[metric] = {'statistic': float('nan'), 'p_value': float('nan'), 'significant': False}
            continue

        stat, p_value = ttest_rel(a[valid], b[valid])
        significant = p_value < 0.05
        results[metric] = {'statistic': stat, 'p_value': p_value, 'significant': significant}

        sig_marker = '* (p<0.05)' if significant else 'ns'
        logger.info(f"  {metric:<12} {stat:>10.4f} {p_value:>10.4f} {sig_marker:>12}")

    return results


def compare_models():
    """Compare Single Ising Model vs Neural Ising Network on all datasets using K-Fold CV."""

    # Initialize
    params = get_env_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Reinitialize logger to write to plots directory BEFORE using it
    global logger
    plotter = Plot(output_dir=f'plots_{run_timestamp}')
    logger = Logger(log_dir=f'plots_{run_timestamp}')
    
    print_config(params, run_timestamp)

    dataset_manager = DatasetManager()

    # Find all datasets
    datasets_dir = os.path.join(os.path.dirname(__file__), 'Datasets')
    csv_files = sorted(glob.glob(os.path.join(datasets_dir, '*.csv')))

    if not csv_files:
        logger.error(f"No CSV files found in {datasets_dir}")
        return

    # Results storage
    dataset_names = []
    single_accuracies_all = []  # List of lists (one list per dataset with k accuracies)
    neural_accuracies_all = []
    single_losses_all = []
    neural_losses_all = []
    all_single_metrics = []  # [{metric: [fold_values]} per dataset]
    all_neural_metrics = []

    total_start = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info(f"\n[Dataset {idx}/{len(csv_files)}: {name}]")

        try:
            # Load dataset
            X, y = dataset_manager.load_csv_dataset(csv_path)

            # Generate K-Folds
            folds = dataset_manager.generate_k_folds(X, y, params['k_folds'])
            logger.info(f"Running {params['k_folds']}-Fold Cross Validation")

            single_accs = []
            neural_accs = []
            single_losses = []
            neural_losses = []
            single_val_accs_all = []
            neural_val_accs_all = []
            single_fold_metrics = {m: [] for m in METRICS}
            neural_fold_metrics = {m: [] for m in METRICS}

            for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds, 1):
                logger.info(f"Fold {fold_idx}/{params['k_folds']}")

                # Train Single Model
                logger.info("  Training Single Model...")
                acc_s, probs_s, loss_s, val_accs_s = train_single_model(X_train, y_train, X_test, y_test, params)
                ms = compute_metrics(y_test, probs_s)
                logger.info(f"  Single | acc={ms['accuracy']:.4f} prec={ms['precision']:.4f} rec={ms['recall']:.4f} f1={ms['f1']:.4f} auc={ms['auc']:.4f}")

                # Train Neural Network
                logger.info("  Training Neural Network...")
                acc_n, probs_n, loss_n, val_accs_n = train_neural_net(X_train, y_train, X_test, y_test, params)
                mn = compute_metrics(y_test, probs_n)
                logger.info(f"  Neural | acc={mn['accuracy']:.4f} prec={mn['precision']:.4f} rec={mn['recall']:.4f} f1={mn['f1']:.4f} auc={mn['auc']:.4f}")

                for k in METRICS:
                    single_fold_metrics[k].append(ms[k])
                    neural_fold_metrics[k].append(mn[k])

                single_accs.append(ms['accuracy'])
                neural_accs.append(mn['accuracy'])
                single_losses.append(loss_s)
                neural_losses.append(loss_n)
                single_val_accs_all.append(val_accs_s)
                neural_val_accs_all.append(val_accs_n)

            # Store results
            dataset_names.append(name)
            single_accuracies_all.append(single_accs)
            neural_accuracies_all.append(neural_accs)
            single_losses_all.append(single_losses)
            neural_losses_all.append(neural_losses)
            all_single_metrics.append(single_fold_metrics)
            all_neural_metrics.append(neural_fold_metrics)

            # Calculate mean accuracies for this dataset
            mean_acc_s = np.mean(single_accs)
            mean_acc_n = np.mean(neural_accs)
            std_acc_s = np.std(single_accs)
            std_acc_n = np.std(neural_accs)

            logger.info(f"\nResults | Single: {mean_acc_s:.4f}±{std_acc_s:.4f} | Neural: {mean_acc_n:.4f}±{std_acc_n:.4f} | Diff: {mean_acc_n - mean_acc_s:+.4f}\n")

            # Paired t-test on same folds
            paired_ttest_kfold(single_fold_metrics, neural_fold_metrics, name)

            # Plot loss/accuracy for this dataset (using first fold)
            plotter.plot_loss_accuracy(single_losses[0], single_val_accs_all[0], f"{name}_single")
            plotter.plot_loss_accuracy(neural_losses[0], neural_val_accs_all[0], f"{name}_neural")

        except Exception as e:
            logger.error(f"ERROR: {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start
    logger.info(f"\n[COMPLETED in {total_time:.1f}s]\n")

    # Generate summary plots
    logger.info("Generating summary plots...")

    # Calculate mean accuracies across folds for each dataset
    single_means = [np.mean(accs) for accs in single_accuracies_all]
    neural_means = [np.mean(accs) for accs in neural_accuracies_all]

    # Plot overall accuracy comparison
    plotter.plot_tot_accuracy(single_means, dataset_names)
    plotter.plot_compare_accuracy(single_means, neural_means, dataset_names, model_name1="FullIsingModule", model_name2="MultiIsingNetwork")
    plotter.plot_parity_scatter(single_means, neural_means, dataset_names, model_name1="FullIsingModule", model_name2="MultiIsingNetwork")
    try:
        # Assumiamo che single_means e neural_means siano già calcolati
        stat, p_value = wilcoxon(single_means, neural_means)
        logger.info(f"\nWilcoxon signed-rank test: statistic={stat:.4f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            logger.info("Differenza significativa tra i modelli (p < 0.05).")
        else:
            logger.info("Nessuna differenza significativa tra i modelli (p ≥ 0.05).")
    except Exception as e:
        logger.error(f"Errore nel test di Wilcoxon: {e}")
    
    # Box plots for both models
    plotter.box_plot(
        single_accuracies_all,
        dataset_names,
        title='Single Ising Model: K-Fold Cross-Validation Accuracy Distribution',
        filename='box_plot_single_model',
        box_color='primary'
    )
    plotter.box_plot(
        neural_accuracies_all,
        dataset_names,
        title='Multi-Ising Neural Network: K-Fold Cross-Validation Accuracy Distribution',
        filename='box_plot_neural_network',
        box_color='secondary'
    )

    # === Save metrics CSV ===
    if dataset_names:
        csv_path = save_metrics_csv(dataset_names, all_single_metrics, all_neural_metrics, plotter.output_dir,
                                     model_name1='FullIsingModule', model_name2='NeuralNet')
        logger.info(f"Metrics CSV saved to: {csv_path}")

        # Prepare per-metric aggregated arrays
        m1 = {m: [np.nanmean(all_single_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}
        m2 = {m: [np.nanmean(all_neural_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}
        e1 = {m: [np.nanstd(all_single_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}
        e2 = {m: [np.nanstd(all_neural_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}

        # DataFrames for heatmap (rename keys to readable labels)
        col_map = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'auc': 'AUC'}
        df_single = pd.DataFrame({col_map[m]: m1[m] for m in METRICS}, index=dataset_names)
        df_neural = pd.DataFrame({col_map[m]: m2[m] for m in METRICS}, index=dataset_names)

        # Means across datasets for radar chart
        means1_radar = {m: np.nanmean(m1[m]) for m in METRICS}
        means2_radar = {m: np.nanmean(m2[m]) for m in METRICS}

        # New summary plots
        plotter.plot_metrics_bar_comparison(
            m1, m2, dataset_names, errors1=e1, errors2=e2,
            model_name1='FullIsingModule', model_name2='NeuralNet'
        )
        plotter.plot_metrics_heatmap(
            df_single, df_neural,
            model_name1='FullIsingModule', model_name2='NeuralNet'
        )
        plotter.plot_radar_chart(
            means1_radar, means2_radar,
            model_name1='FullIsingModule', model_name2='NeuralNet'
        )
        plotter.plot_combined_boxplot(
            [all_single_metrics[i]['accuracy'] for i in range(len(dataset_names))],
            [all_neural_metrics[i]['accuracy'] for i in range(len(dataset_names))],
            dataset_names,
            model_name1='FullIsingModule', model_name2='NeuralNet'
        )

    logger.info(f"All plots saved to {plotter.output_dir}")


def iris_matrix():
    """
    Matrix experiment on Iris (versicolor vs virginica):
      rows = num_ising_perceptrons  (NUM_NODI_LIST)
      cols = size_annealer          (_node_size_configs)
    Single train/test split, no K-fold.
    """
    import traceback
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path
    from time import perf_counter
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset
    from Inference.test_matrix_xor import (
        NUM_NODI_LIST, _node_size_configs,
        _heatmap, _timing_heatmap, _plot_roc_curve,
    )

    params = get_env_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"plots_iris_matrix_{run_timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    local_logger = Logger(log_dir=str(output_dir))

    # ── load iris ──────────────────────────────────────────────────────────────
    iris_path = os.path.join(os.path.dirname(__file__), 'Datasets', '00_iris_versicolor_virginica.csv')
    X, y = DatasetManager().load_csv_dataset(iris_path)

    # ── single train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=params['random_seed'],
        stratify=y,
    )
    # standardize (fit on train only)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    num_features = X_train.shape[1]
    node_configs = _node_size_configs(num_features)
    col_labels   = [lbl for _, lbl in node_configs]
    plot_col_labels = [
        f"{lbl}\n({size} nodes)" if lbl.endswith('F') else lbl
        for size, lbl in node_configs
    ]
    row_labels   = [str(n) for n in NUM_NODI_LIST]
    n_rows       = len(NUM_NODI_LIST)
    n_cols       = len(node_configs)

    local_logger.info(f"\n{'='*80}")
    local_logger.info(f"  IRIS MATRIX EXPERIMENT  [{datetime.now().strftime('%H:%M:%S')}]")
    local_logger.info(f"{'='*80}")
    local_logger.info(f"  Train: {len(y_train)} samples  |  Test: {len(y_test)} samples")
    local_logger.info(f"  Grid : {n_rows} × {n_cols} = {n_rows * n_cols} configurations")

    # ── result matrices ───────────────────────────────────────────────────────
    acc_matrix       = np.full((n_rows, n_cols), np.nan)
    f1_matrix        = np.full((n_rows, n_cols), np.nan)
    auc_matrix       = np.full((n_rows, n_cols), np.nan)
    time_matrix      = np.full((n_rows, n_cols), np.nan)
    best_val_matrix  = np.full((n_rows, n_cols), np.nan)
    detailed_records = []
    total_configs    = n_rows * n_cols
    done             = 0

    # ── annealing settings (shared across all configs) ────────────────────────
    SA_settings                     = AnnealingSettings()
    SA_settings.beta_range          = params['sa_beta_range']
    SA_settings.num_reads           = params['sa_num_reads']
    SA_settings.num_sweeps          = params['sa_num_sweeps']
    SA_settings.num_sweeps_per_beta = params['sa_sweeps_per_beta']

    val_interval = int(os.getenv('VALIDATION_INTERVAL', 5))
    exp_start    = perf_counter()

    for i, num_nodi in enumerate(NUM_NODI_LIST):
        for j, (node_size, node_label) in enumerate(node_configs):
            done += 1
            local_logger.info(
                f"\n  [{done:3d}/{total_configs}]  "
                f"num_nodi={num_nodi:3d}  node_size={node_size:3d} ({node_label:4s})"
            )
            try:
                t0  = perf_counter()
                dev = params['device']
                torch.manual_seed(params['random_seed'])
                np.random.seed(params['random_seed'])

                # data tensors
                tx = torch.tensor(X_train, dtype=torch.float32)
                ty = torch.tensor(y_train, dtype=torch.float32)
                train_loader = DataLoader(
                    TensorDataset(tx, ty),
                    batch_size=params['batch_size'],
                    shuffle=True,
                )
                ex = torch.tensor(X_test, dtype=torch.float32)

                # model
                model = MultiIsingNetwork(
                    num_ising_perceptrons=num_nodi,
                    size_annealer=node_size,
                    annealing_settings=SA_settings,
                    annealer_type=AnnealerType.SIMULATED,
                    lambda_init=params['lambda_init'],
                    offset_init=params['offset_init'],
                    partition_input=False,
                ).to(dev)

                n_params = sum(p.numel() for p in model.parameters())

                # grouped optimizer
                optim_groups = []
                for module in model.ising_perceptrons_layer:
                    optim_groups += [
                        {'params': [module.gamma],  'lr': params['lr_gamma']},
                        {'params': [module.lmd],    'lr': params['lr_lambda']},
                        {'params': [module.offset], 'lr': params['lr_offset']},
                    ]
                optim_groups.append({
                    'params': model.combiner_layer.parameters(),
                    'lr': params['lr_combiner'],
                })
                optimizer = torch.optim.Adam(optim_groups)
                loss_fn   = torch.nn.BCEWithLogitsLoss()

                # training loop
                training_losses = []
                val_accuracies  = []
                best_val_acc    = 0.0

                for epoch in range(params['epochs']):
                    model.train()
                    batch_losses = []
                    for xb, yb in train_loader:
                        xb = xb.to(dev)
                        yb = yb.to(dev).float()
                        optimizer.zero_grad()
                        loss = loss_fn(flatten_logits(model(xb)), yb)
                        loss.backward()
                        optimizer.step()
                        batch_losses.append(loss.item())

                    avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
                    training_losses.append(avg_loss)

                    do_val = (
                        (epoch + 1) % val_interval == 0
                        or epoch == 0
                        or epoch == params['epochs'] - 1
                    )
                    if do_val:
                        model.eval()
                        with torch.no_grad():
                            logits    = flatten_logits(model(ex.to(dev)))
                            probs_val = torch.sigmoid(logits).cpu().numpy()
                            val_acc   = accuracy_score(y_test, (probs_val >= 0.5).astype(int))
                        val_accuracies.append(val_acc)
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                    else:
                        val_accuracies.append(val_accuracies[-1] if val_accuracies else 0.0)

                # final evaluation
                model.eval()
                with torch.no_grad():
                    logits = flatten_logits(model(ex.to(dev)))
                    probs  = torch.sigmoid(logits).cpu().numpy()
                preds     = (probs >= 0.5).astype(int)
                final_acc = accuracy_score(y_test, preds)
                f1        = f1_score(y_test, preds, average='binary', zero_division=0)
                try:
                    auc = roc_auc_score(y_test, probs)
                except Exception:
                    auc = float('nan')

                final_loss = training_losses[-1] if training_losses else float('nan')
                min_loss   = min(training_losses)  if training_losses else float('nan')
                elapsed    = perf_counter() - t0

                acc_matrix[i, j]      = final_acc
                f1_matrix[i, j]       = f1
                auc_matrix[i, j]      = auc
                time_matrix[i, j]     = elapsed
                best_val_matrix[i, j] = best_val_acc

                local_logger.info(
                    f"    acc={final_acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
                    f"best_val={best_val_acc:.4f}  min_loss={min_loss:.4f}  "
                    f"final_loss={final_loss:.4f}  time={elapsed:.1f}s  params={n_params}"
                )

                if not np.isnan(auc):
                    _plot_roc_curve(
                        y_test, probs, auc,
                        f'ROC — nodi={num_nodi} size={node_size} (Iris)',
                        output_dir / f'roc_nodi{num_nodi}_size{node_label}.png',
                    )

                detailed_records.append({
                    'num_nodi':         num_nodi,
                    'node_size':        node_size,
                    'node_size_label':  node_label,
                    'accuracy':         final_acc,
                    'f1':               f1,
                    'auc_roc':          auc,
                    'best_val_acc':     best_val_acc,
                    'final_loss':       final_loss,
                    'min_loss':         min_loss,
                    'training_time_s':  elapsed,
                    'n_params':         n_params,
                    'error':            '',
                })

            except Exception as e:
                local_logger.error(f"    ERROR: {e}")
                traceback.print_exc()
                detailed_records.append({
                    'num_nodi': num_nodi, 'node_size': node_size,
                    'node_size_label': node_label,
                    'accuracy': np.nan, 'f1': np.nan, 'auc_roc': np.nan,
                    'best_val_acc': np.nan, 'final_loss': np.nan,
                    'min_loss': np.nan, 'training_time_s': np.nan,
                    'n_params': np.nan, 'error': str(e),
                })

    exp_elapsed = perf_counter() - exp_start
    local_logger.info(f"\n  Iris grid completed in {exp_elapsed:.1f}s ({exp_elapsed/60:.1f} min)")

    # ── save CSV matrices ──────────────────────────────────────────────────────
    def _save_mat(mat, name):
        df = pd.DataFrame(mat, index=row_labels, columns=col_labels)
        df.index.name = 'num_nodi'
        df.to_csv(output_dir / f"{name}.csv")
        return df

    df_acc  = _save_mat(acc_matrix,      'accuracy_matrix')
    _save_mat(f1_matrix,       'f1_matrix')
    _save_mat(auc_matrix,      'auc_matrix')
    _save_mat(time_matrix,     'timing_matrix')
    _save_mat(best_val_matrix, 'best_val_matrix')
    pd.DataFrame(detailed_records).to_csv(output_dir / 'detailed_metrics.csv', index=False)

    # ── heatmaps ───────────────────────────────────────────────────────────────
    _heatmap(acc_matrix,      row_labels, plot_col_labels,
             'Accuracy — Iris',       output_dir / 'accuracy_heatmap.png',
             cmap='YlOrRd', cbar_label='Accuracy')
    _heatmap(f1_matrix,       row_labels, plot_col_labels,
             'F1-Score — Iris',       output_dir / 'f1_heatmap.png',
             cmap='Blues',  cbar_label='F1-Score')
    _heatmap(auc_matrix,      row_labels, plot_col_labels,
             'AUC-ROC — Iris',        output_dir / 'auc_heatmap.png',
             cmap='Greens', cbar_label='AUC-ROC')
    _heatmap(best_val_matrix, row_labels, plot_col_labels,
             'Best Val Acc — Iris',   output_dir / 'best_val_heatmap.png',
             cmap='Purples', cbar_label='Best Val Acc')
    _timing_heatmap(time_matrix, row_labels, plot_col_labels,
                    'Training Time (s) — Iris', output_dir / 'timing_heatmap.png')

    # ── console summary ────────────────────────────────────────────────────────
    local_logger.info(f"\n{'─'*70}")
    local_logger.info("  ACCURACY MATRIX  (rows=num_nodi, cols=node_size)  IRIS")
    local_logger.info(f"{'─'*70}")
    local_logger.info(df_acc.to_string())

    if not np.all(np.isnan(acc_matrix)):
        best_idx = np.unravel_index(np.nanargmax(acc_matrix), acc_matrix.shape)
        local_logger.info(
            f"\n  BEST CONFIG:  num_nodi={NUM_NODI_LIST[best_idx[0]]}  "
            f"node_size={col_labels[best_idx[1]]}  "
            f"→  acc={acc_matrix[best_idx]:.4f}"
        )
    local_logger.info(f"  Output dir: {output_dir.resolve()}")


if __name__ == '__main__':

    # Compare Single vs Multi Ising on all datasets
    #compare_models()

    # Iris matrix experiment (num_nodi × node_size), no K-fold
    iris_matrix()

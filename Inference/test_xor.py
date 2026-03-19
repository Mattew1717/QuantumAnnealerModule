"""
Test XOR on Network_2L (TwoStageIsingNetwork) from 1D to 6D
Tests XOR problem across multiple dimensions, trains Network_2L, validates, and saves confusion matrices.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from time import perf_counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Add repository root to sys.path to enable absolute imports
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.utils.logger import Logger
from Inference.utils.plot import Plot
from Inference.utils.utils import generate_xor_balanced
from Inference.utils.dataset_manager import  HiddenNodesInitialization, SimpleDataset
from ModularNetwork.Network_2L import TwoLayerIsingNetwork
from ModularNetwork.Network_1L import MultiIsingNetwork
from full_ising_model.full_ising_module import FullIsingModule
from full_ising_model.annealers import AnnealingSettings, AnnealerType
from torch.utils.data import DataLoader, TensorDataset

logger = Logger()
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    """Safely flatten model output to 1D per-sample logits."""
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(1)
    raise ValueError(f"Unexpected model output shape: {logits.shape}")


def compute_metrics(y_true, probs):
    """Compute accuracy, precision, recall, F1, AUC from predicted probabilities."""
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float('nan')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


def _save_metrics_csv(dataset_names, all_single_metrics, all_neural_metrics, output_dir):
    """Save per-dimension metrics summary to CSV (single split, no K-Fold)."""
    rows = []
    for i, ds_name in enumerate(dataset_names):
        for model_name, model_metrics in [('FullIsingModule', all_single_metrics[i]), ('Network_1L', all_neural_metrics[i])]:
            row = {'dataset': ds_name, 'model': model_name}
            for m in METRICS:
                vals = model_metrics[m]
                row[f'{m}_mean'] = np.nanmean(vals)
                row[f'{m}_std'] = np.nanstd(vals)
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(str(output_dir), 'metrics_summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    return csv_path


def get_xor_params():
    """Load parameters for XOR experiments."""
    return {
        'random_seed': int(os.getenv('RANDOM_SEED')),
        'batch_size': int(os.getenv('BATCH_SIZE')),
        'model_size': int(os.getenv('MODEL_SIZE')),
        'num_ising_1': int(os.getenv('NUM_ISING_1')),  
        'num_ising_2': int(os.getenv('NUM_ISING_2')),  
        'num_ising_net': int(os.getenv('NUM_ISING_PERCEPTRONS')),
        'epochs': int(os.getenv('EPOCHS')),
        'lambda_init': float(os.getenv('LAMBDA_INIT')),
        'offset_init': float(os.getenv('OFFSET_INIT')),
        'lr_gamma': float(os.getenv('LEARNING_RATE_GAMMA')),
        'lr_lambda': float(os.getenv('LEARNING_RATE_LAMBDA')),
        'lr_offset': float(os.getenv('LEARNING_RATE_OFFSET')),
        'lr_classical': float(os.getenv('LEARNING_RATE_COMBINER')),
        'sa_beta_range': [int(os.getenv('SA_BETA_MIN')), int(os.getenv('SA_BETA_MAX'))],
        'num_reads': int(os.getenv('NUM_READS')),
        'sa_num_sweeps': int(os.getenv('SA_NUM_SWEEPS')),
        'sa_sweeps_per_beta': int(os.getenv('SA_SWEEPS_PER_BETA')),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_samples_per_region': int(os.getenv('N_SAMPLES_PER_REGION')),  
        'test_size': float(os.getenv('TEST_SIZE')), 
        'num_workers': int(os.getenv("NUM_THREADS"))
    }


def prepare_xor_data(dim, params):
    """Generate and prepare XOR data for given dimension."""
    logger.info(f"Generating {dim}D XOR data...")

    # Generate balanced XOR data
    X, y = generate_xor_balanced(
        dim=dim,
        n_samples_dim=params['n_samples_per_region'],
        shuffle=True,
        random_seed=params['random_seed']
    )

    logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

    # Train/test split with stratification to ensure balanced distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['test_size'],
        random_state=params['random_seed'],
        stratify=y
    )

    # Verify balanced split
    train_class_dist = np.bincount(y_train.astype(int))
    test_class_dist = np.bincount(y_test.astype(int))

    logger.info(f"Train set: {len(y_train)} samples - Class distribution: {train_class_dist} ({train_class_dist/len(y_train)*100}%)")
    logger.info(f"Test set: {len(y_test)} samples - Class distribution: {test_class_dist} ({test_class_dist/len(y_test)*100}%)")

    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, model_size, params):
    """Create PyTorch dataloaders with standardization and hidden nodes."""

    # Standardization (fit on train only, no leakage)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Create SimpleDataset objects
    train_dataset = SimpleDataset()
    train_dataset.x = torch.tensor(X_train, dtype=torch.float32)
    train_dataset.y = torch.tensor(y_train, dtype=torch.float32)
    train_dataset.data_size = X_train.shape[1]
    train_dataset.len = len(y_train)

    test_dataset = SimpleDataset()
    test_dataset.x = torch.tensor(X_test, dtype=torch.float32)
    test_dataset.y = torch.tensor(y_test, dtype=torch.float32)
    test_dataset.data_size = X_test.shape[1]
    test_dataset.len = len(y_test)

    # No manual resize needed - FullIsingModule handles it automatically in forward()

    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(train_dataset.x, train_dataset.y),
        batch_size=params['batch_size'],
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(test_dataset.x, test_dataset.y),
        batch_size=params['batch_size'],
        shuffle=False
    )

    return train_dataset, test_dataset, train_loader, test_loader


def train_network_2L(dim, X_train, y_train, X_test, y_test, params, plotter):
    """Train TwoStageIsingNetwork on XOR data."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Network_2L on {dim}D XOR")
    logger.info(f"{'='*60}")

    # Determine model size
    if params['model_size'] == -1:
        min_size = int(os.getenv('MINIMUM_MODEL_SIZE', 20))
        model_size = max(X_train.shape[1], min_size)
    else:
        model_size = params['model_size']

    logger.info(f"Model size: {model_size}")

    # Prepare data
    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, model_size, params
    )

    # Setup annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.num_sweeps_per_beta = params['sa_sweeps_per_beta']

    # Create Network_2L
    model = TwoLayerIsingNetwork(
        input_dim=model_size,
        num_ising_1=params['num_ising_1'],
        num_ising_2=params['num_ising_2'],
        annealing_settings=SA_settings,
        annealer_type=AnnealerType.SIMULATED,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        bias=True
    ).to(params['device'])

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup optimizer with grouped learning rates
    optimizer_groups = []

    # First Ising layer parameters
    for module in model.ising1:
        optimizer_groups.append({
            'params': [module.gamma],
            'lr': params['lr_gamma']
        })
        optimizer_groups.append({
            'params': [module.lmd],
            'lr': params['lr_lambda']
        })
        optimizer_groups.append({
            'params': [module.offset],
            'lr': params['lr_offset']
        })

    # Second Ising layer parameters
    for module in model.ising2:
        optimizer_groups.append({
            'params': [module.gamma],
            'lr': params['lr_gamma']
        })
        optimizer_groups.append({
            'params': [module.lmd],
            'lr': params['lr_lambda']
        })
        optimizer_groups.append({
            'params': [module.offset],
            'lr': params['lr_offset']
        })

    # Classical layers (lin1, lin2)
    optimizer_groups.append({
        'params': list(model.lin1.parameters()) + list(model.lin2.parameters()),
        'lr': params['lr_classical']
    })

    optimizer = torch.optim.Adam(optimizer_groups)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    logger.info("\nStarting training...")
    training_losses = []
    validation_accuracies = []
    val_interval = int(os.getenv('VALIDATION_INTERVAL', 5))  # Validate every N epochs

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(params['device'])
            y_batch = y_batch.to(params['device']).float()

            optimizer.zero_grad()
            pred = _flatten_logits(model(x_batch))
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)

        # Validation - only every val_interval epochs to save time
        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == params['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                preds_tensor = _flatten_logits(model(test_dataset.x.to(params['device'])))
                probs = torch.sigmoid(preds_tensor).cpu().numpy()
                predictions = np.where(probs < 0.5, 0, 1)
                epoch_accuracy = accuracy_score(y_test, predictions)
                validation_accuracies.append(epoch_accuracy)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {epoch_accuracy:.4f}")
        else:
            # Append last known accuracy for plotting
            if validation_accuracies:
                validation_accuracies.append(validation_accuracies[-1])

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f}")

    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation")
    logger.info("="*60)

    model.eval()
    with torch.no_grad():
        preds_tensor = _flatten_logits(model(test_dataset.x.to(params['device'])))
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
        predictions = np.where(probs < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1'], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

    # Plot loss and accuracy curves
    plotter.plot_loss_accuracy(training_losses, validation_accuracies, f"XOR_{dim}D_Network2L")

    # Plot confusion matrix
    plotter.plot_confusion_matrix(y_test, predictions, labels=[0, 1], filename=f"confusion_matrix_xor_{dim}d")

    return final_accuracy, training_losses, validation_accuracies, predictions, y_test, probs


def train_network_1L(dim, X_train, y_train, X_test, y_test, params, plotter):
    """Train MultiIsingNetwork (1L) on XOR data."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Network_1L on {dim}D XOR")
    logger.info(f"{'='*60}")

    # Determine model size
    if params['model_size'] == -1:
        min_size = int(os.getenv('MINIMUM_MODEL_SIZE', 20))
        model_size = max(X_train.shape[1], min_size)
    else:
        model_size = params['model_size']

    logger.info(f"Model size: {model_size}")

    # Prepare data
    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, model_size, params
    )

    # Setup annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.num_sweeps_per_beta = params['sa_sweeps_per_beta']

    # Create Network_1L (MultiIsingNetwork)
    model = MultiIsingNetwork(
        num_ising_perceptrons=params['num_ising_net'],  # Use num_ising_net as the number of perceptrons
        size_annealer=model_size,
        annealing_settings=SA_settings,
        annealer_type=AnnealerType.SIMULATED,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        partition_input=False  # No partitioning for fair comparison
    ).to(params['device'])

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup optimizer with grouped learning rates
    optimizer_groups = []

    # Ising layer parameters
    for module in model.ising_perceptrons_layer:
        optimizer_groups.append({
            'params': [module.gamma],
            'lr': params['lr_gamma']
        })
        optimizer_groups.append({
            'params': [module.lmd],
            'lr': params['lr_lambda']
        })
        optimizer_groups.append({
            'params': [module.offset],
            'lr': params['lr_offset']
        })

    # Combiner layer
    optimizer_groups.append({
        'params': model.combiner_layer.parameters(),
        'lr': params['lr_classical']
    })

    optimizer = torch.optim.Adam(optimizer_groups)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    logger.info("\nStarting training...")
    training_losses = []
    validation_accuracies = []
    val_interval = int(os.getenv('VALIDATION_INTERVAL', 5))  # Validate every N epochs

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(params['device'])
            y_batch = y_batch.to(params['device']).float()

            optimizer.zero_grad()
            pred = _flatten_logits(model(x_batch))
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)

        # Validation - only every val_interval epochs to save time
        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == params['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                preds_tensor = _flatten_logits(model(test_dataset.x.to(params['device'])))
                probs = torch.sigmoid(preds_tensor).cpu().numpy()
                predictions = np.where(probs < 0.5, 0, 1)
                epoch_accuracy = accuracy_score(y_test, predictions)
                validation_accuracies.append(epoch_accuracy)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {epoch_accuracy:.4f}")
        else:
            # Append last known accuracy for plotting
            if validation_accuracies:
                validation_accuracies.append(validation_accuracies[-1])

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f}")

    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation")
    logger.info("="*60)

    model.eval()
    with torch.no_grad():
        preds_tensor = _flatten_logits(model(test_dataset.x.to(params['device'])))
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
        predictions = np.where(probs < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1'], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

    # Plot loss and accuracy curves
    plotter.plot_loss_accuracy(training_losses, validation_accuracies, f"XOR_{dim}D_Network1L")

    # Plot confusion matrix
    plotter.plot_confusion_matrix(y_test, predictions, labels=[0, 1], filename=f"confusion_matrix_xor_{dim}d_1L")

    return final_accuracy, training_losses, validation_accuracies, predictions, y_test, probs


def train_full_ising_model(dim, X_train, y_train, X_test, y_test, params, plotter):
    """Train FullIsingModule on XOR data."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Training FullIsingModule on {dim}D XOR")
    logger.info(f"{'='*60}")

    if params['model_size'] == -1:
        min_size = int(os.getenv('MINIMUM_MODEL_SIZE', 20))
        model_size = max(X_train.shape[1], min_size)
    else:
        model_size = params['model_size']

    logger.info(f"Model size: {model_size}")

    train_dataset, test_dataset, train_loader, _ = create_dataloaders(
        X_train, y_train, X_test, y_test, model_size, params
    )

    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.num_sweeps_per_beta = params['sa_sweeps_per_beta']

    model = FullIsingModule(
        size_annealer=model_size,
        annealer_type=AnnealerType.SIMULATED,
        annealing_settings=SA_settings,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init']
    ).to(params['device'])

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = torch.optim.Adam([
        {'params': [model.gamma], 'lr': params['lr_gamma']},
        {'params': [model.lmd], 'lr': params['lr_lambda']},
        {'params': [model.offset], 'lr': params['lr_offset']},
    ])

    loss_fn = torch.nn.BCEWithLogitsLoss()

    logger.info("\nStarting training...")
    training_losses = []
    validation_accuracies = []
    val_interval = int(os.getenv('VALIDATION_INTERVAL', 5))

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(params['device'])
            y_batch = y_batch.to(params['device']).float()

            optimizer.zero_grad()
            pred = _flatten_logits(model(x_batch))
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)

        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == params['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                preds_tensor = _flatten_logits(model(test_dataset.x.to(params['device'])))
                probs = torch.sigmoid(preds_tensor).cpu().numpy()
                predictions = np.where(probs < 0.5, 0, 1)
                epoch_accuracy = accuracy_score(y_test, predictions)
                validation_accuracies.append(epoch_accuracy)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {epoch_accuracy:.4f}")
        else:
            if validation_accuracies:
                validation_accuracies.append(validation_accuracies[-1])

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f}")

    logger.info("\n" + "="*60)
    logger.info("Final Evaluation")
    logger.info("="*60)

    model.eval()
    with torch.no_grad():
        preds_tensor = _flatten_logits(model(test_dataset.x.to(params['device'])))
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
        predictions = np.where(probs < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1'], zero_division=0))

    cm = confusion_matrix(y_test, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

    plotter.plot_loss_accuracy(training_losses, validation_accuracies, f"XOR_{dim}D_FullIsingModule")
    plotter.plot_confusion_matrix(y_test, predictions, labels=[0, 1], filename=f"confusion_matrix_xor_{dim}d_fullising")

    return final_accuracy, training_losses, validation_accuracies, predictions, y_test, probs


def test_xor_all_dimensions():
    """Test XOR from 1D to 6D on Network_2L."""

    # Initialize
    params = get_xor_params()
    
    # Create output directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter = Plot(output_dir=f'plots_xor_{run_timestamp}')
    
    # Reinitialize logger to write to plots directory
    global logger
    logger = Logger(log_dir=f'plots_xor_{run_timestamp}')

    logger.info("\n" + "="*80)
    logger.info("XOR Test Suite for Network_2L (TwoStageIsingNetwork)")
    logger.info("="*80)
    logger.info(f"Device: {params['device']}")
    logger.info(f"Batch size: {params['batch_size']}")
    logger.info(f"Epochs: {params['epochs']}")
    logger.info(f"Model size: {params['model_size']}")
    logger.info(f"Learning rates: gamma={params['lr_gamma']}, lambda={params['lr_lambda']}, offset={params['lr_offset']}, classical={params['lr_classical']}")
    logger.info(f"Annealing: beta={params['sa_beta_range']}, reads={params['num_reads']}, sweeps={params['sa_num_sweeps']}")

    # Store results
    results = {
        'dimensions': [],
        'accuracies': [],
        'losses': [],
        'val_accuracies': [],
        'execution_times': []
    }

    # Start total timer
    total_start = perf_counter()

    # Test each dimension
    for dim in range(1, 7):
        try:
            # Start timer for this dimension
            dim_start = perf_counter()

            # Generate data
            X_train, X_test, y_train, y_test = prepare_xor_data(dim, params)

            # Train and evaluate
            accuracy, losses, val_accs, predictions, y_true, _ = train_network_2L(
                dim, X_train, y_train, X_test, y_test, params, plotter
            )

            # End timer for this dimension
            dim_time = perf_counter() - dim_start

            # Store results
            results['dimensions'].append(dim)
            results['accuracies'].append(accuracy)
            results['losses'].append(losses)
            results['val_accuracies'].append(val_accs)
            results['execution_times'].append(dim_time)

            logger.info(f"\n{dim}D XOR completed with accuracy: {accuracy:.4f} | Execution time: {dim_time:.2f}s\n")

        except Exception as e:
            logger.error(f"Error on {dim}D XOR: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # End total timer
    total_time = perf_counter() - total_start

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - XOR Performance Across Dimensions")
    logger.info("="*80)

    for dim, acc, exec_time in zip(results['dimensions'], results['accuracies'], results['execution_times']):
        logger.info(f"{dim}D XOR: {acc:.4f} | Execution time: {exec_time:.2f}s")

    logger.info(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    # Plot summary
    logger.info(f"\nPlots saved to {plotter.output_dir}")

    # Create dimension comparison plot
    plotter.plot_tot_accuracy(results['accuracies'], [f"{d}D" for d in results['dimensions']])

    logger.info("\nXOR Test Suite Completed!")


def test_xor_all_dimensions_1L():
    """Test XOR from 1D to 6D on Network_1L (MultiIsingNetwork)."""

    # Initialize
    params = get_xor_params()

    # Create output directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter = Plot(output_dir=f'plots_xor_1L_{run_timestamp}')

    # Reinitialize logger to write to plots directory
    global logger
    logger = Logger(log_dir=f'plots_xor_1L_{run_timestamp}')

    logger.info("\n" + "="*80)
    logger.info("XOR Test Suite for Network_1L (MultiIsingNetwork)")
    logger.info("="*80)
    logger.info(f"Device: {params['device']}")
    logger.info(f"Batch size: {params['batch_size']}")
    logger.info(f"Epochs: {params['epochs']}")
    logger.info(f"Model size: {params['model_size']}")
    logger.info(f"Num Ising Perceptrons: {params['num_ising_net']}")
    logger.info(f"Learning rates: gamma={params['lr_gamma']}, lambda={params['lr_lambda']}, offset={params['lr_offset']}, classical={params['lr_classical']}")
    logger.info(f"Annealing: beta={params['sa_beta_range']}, reads={params['num_reads']}, sweeps={params['sa_num_sweeps']}")

    # Store results
    results = {
        'dimensions': [],
        'accuracies': [],
        'losses': [],
        'val_accuracies': [],
        'execution_times': []
    }

    # Start total timer
    total_start = perf_counter()

    # Test each dimension
    for dim in range(1, 7):
        try:
            # Start timer for this dimension
            dim_start = perf_counter()

            # Generate data
            X_train, X_test, y_train, y_test = prepare_xor_data(dim, params)

            # Train and evaluate
            accuracy, losses, val_accs, predictions, y_true, _ = train_network_1L(
                dim, X_train, y_train, X_test, y_test, params, plotter
            )

            # End timer for this dimension
            dim_time = perf_counter() - dim_start

            # Store results
            results['dimensions'].append(dim)
            results['accuracies'].append(accuracy)
            results['losses'].append(losses)
            results['val_accuracies'].append(val_accs)
            results['execution_times'].append(dim_time)

            logger.info(f"\n{dim}D XOR completed with accuracy: {accuracy:.4f} | Execution time: {dim_time:.2f}s\n")

        except Exception as e:
            logger.error(f"Error on {dim}D XOR: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # End total timer
    total_time = perf_counter() - total_start

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - XOR Performance Across Dimensions (Network_1L)")
    logger.info("="*80)

    for dim, acc, exec_time in zip(results['dimensions'], results['accuracies'], results['execution_times']):
        logger.info(f"{dim}D XOR: {acc:.4f} | Execution time: {exec_time:.2f}s")

    logger.info(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    # Plot summary
    logger.info(f"\nPlots saved to {plotter.output_dir}")

    # Create dimension comparison plot
    plotter.plot_tot_accuracy(results['accuracies'], [f"{d}D" for d in results['dimensions']])

    logger.info("\nXOR Test Suite for Network_1L Completed!")


def compare_xor_models_all_dimensions():
    """Compare FullIsingModule vs Network_1L on XOR dimensions 1D-6D (single split, no K-Fold)."""

    params = get_xor_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    global logger
    plotter = Plot(output_dir=f'plots_xor_compare_{run_timestamp}')
    logger = Logger(log_dir=f'plots_xor_compare_{run_timestamp}')

    logger.info("\n" + "=" * 80)
    logger.info("XOR Comparison Suite - FullIsingModule vs Network_1L (No K-Fold)")
    logger.info("=" * 80)
    logger.info(f"Device: {params['device']}")
    logger.info(f"Batch size: {params['batch_size']} | Epochs: {params['epochs']} | Model size: {params['model_size']}")
    logger.info(f"Learning rates: gamma={params['lr_gamma']}, lambda={params['lr_lambda']}, offset={params['lr_offset']}, classical={params['lr_classical']}")
    logger.info(f"Annealing: beta={params['sa_beta_range']}, reads={params['num_reads']}, sweeps={params['sa_num_sweeps']}")

    dataset_names = []
    acc_single_all = []
    acc_1l_all = []
    all_single_metrics = []
    all_1l_metrics = []

    total_start = perf_counter()

    for dim in range(1, 7):
        try:
            ds_name = f"{dim}D"
            logger.info(f"\n[Dimension {ds_name}]")

            X_train, X_test, y_train, y_test = prepare_xor_data(dim, params)

            logger.info("  Training FullIsingModule...")
            acc_single, loss_single, val_accs_single, _, y_true_single, probs_single = train_full_ising_model(
                dim, X_train, y_train, X_test, y_test, params, plotter
            )
            ms = compute_metrics(y_true_single, probs_single)
            logger.info(
                f"  FullIsingModule | acc={ms['accuracy']:.4f} prec={ms['precision']:.4f} "
                f"rec={ms['recall']:.4f} f1={ms['f1']:.4f} auc={ms['auc']:.4f}"
            )

            logger.info("  Training Network_1L...")
            acc_1l, loss_1l, val_accs_1l, _, y_true_1l, probs_1l = train_network_1L(
                dim, X_train, y_train, X_test, y_test, params, plotter
            )
            mn = compute_metrics(y_true_1l, probs_1l)
            logger.info(
                f"  Network_1L | acc={mn['accuracy']:.4f} prec={mn['precision']:.4f} "
                f"rec={mn['recall']:.4f} f1={mn['f1']:.4f} auc={mn['auc']:.4f}"
            )

            dataset_names.append(ds_name)
            acc_single_all.append([ms['accuracy']])
            acc_1l_all.append([mn['accuracy']])
            all_single_metrics.append({m: [ms[m]] for m in METRICS})
            all_1l_metrics.append({m: [mn[m]] for m in METRICS})

            logger.info(
                f"  Results | FullIsingModule: {ms['accuracy']:.4f} | Network_1L: {mn['accuracy']:.4f} "
                f"| Diff(1L-Single): {mn['accuracy'] - ms['accuracy']:+.4f}"
            )

        except Exception as e:
            logger.error(f"ERROR on {dim}D XOR comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = perf_counter() - total_start
    logger.info(f"\n[COMPLETED in {total_time:.1f}s]\n")

    if not dataset_names:
        logger.error("No valid XOR dimensions processed. Skipping summary plots.")
        return

    logger.info("Generating summary plots...")

    means_single = [np.nanmean(accs) for accs in acc_single_all]
    means_1l = [np.nanmean(accs) for accs in acc_1l_all]

    plotter.plot_tot_accuracy(means_single, dataset_names)
    plotter.plot_compare_accuracy(means_single, means_1l, dataset_names, model_name1="FullIsingModule", model_name2="Network_1L")
    plotter.plot_parity_scatter(means_single, means_1l, dataset_names, model_name1="FullIsingModule", model_name2="Network_1L")

    try:
        stat, p_value = wilcoxon(means_single, means_1l)
        logger.info(f"\nWilcoxon signed-rank test: statistic={stat:.4f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            logger.info("Differenza significativa tra i modelli (p < 0.05).")
        else:
            logger.info("Nessuna differenza significativa tra i modelli (p >= 0.05).")
    except Exception as e:
        logger.error(f"Errore nel test di Wilcoxon: {e}")

    plotter.box_plot(
        acc_single_all,
        dataset_names,
        title='FullIsingModule: Accuracy Distribution by XOR Dimension',
        filename='box_plot_fullising',
        box_color='primary'
    )
    plotter.box_plot(
        acc_1l_all,
        dataset_names,
        title='Network_1L: Accuracy Distribution by XOR Dimension',
        filename='box_plot_network1l',
        box_color='secondary'
    )

    csv_path = _save_metrics_csv(dataset_names, all_single_metrics, all_1l_metrics, plotter.output_dir)
    logger.info(f"Metrics CSV saved to: {csv_path}")

    ms = {m: [np.nanmean(all_single_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}
    m1l = {m: [np.nanmean(all_1l_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}
    es = {m: [np.nanstd(all_single_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}
    e1l = {m: [np.nanstd(all_1l_metrics[i][m]) for i in range(len(dataset_names))] for m in METRICS}

    col_map = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'auc': 'AUC'}
    df_single = pd.DataFrame({col_map[m]: ms[m] for m in METRICS}, index=dataset_names)
    df_1l = pd.DataFrame({col_map[m]: m1l[m] for m in METRICS}, index=dataset_names)

    means_single_radar = {m: np.nanmean(ms[m]) for m in METRICS}
    means_1l_radar = {m: np.nanmean(m1l[m]) for m in METRICS}

    plotter.plot_metrics_bar_comparison(
        ms, m1l, dataset_names, errors1=es, errors2=e1l,
        model_name1='FullIsingModule', model_name2='Network_1L'
    )
    plotter.plot_metrics_heatmap(
        df_single, df_1l,
        model_name1='FullIsingModule', model_name2='Network_1L'
    )
    plotter.plot_radar_chart(
        means_single_radar, means_1l_radar,
        model_name1='FullIsingModule', model_name2='Network_1L'
    )
    plotter.plot_combined_boxplot(
        [all_single_metrics[i]['accuracy'] for i in range(len(dataset_names))],
        [all_1l_metrics[i]['accuracy'] for i in range(len(dataset_names))],
        dataset_names,
        model_name1='FullIsingModule', model_name2='Network_1L'
    )

    logger.info(f"All plots saved to {plotter.output_dir}")


def fullIsing_xor_all_dim():
    """Test FullIsingModule on XOR from 1D to 6D, plot results and logs."""

    params = get_xor_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    global logger
    plotter = Plot(output_dir=f'plots_xor_fullising_{run_timestamp}')
    logger = Logger(log_dir=f'plots_xor_fullising_{run_timestamp}')

    logger.info("\n" + "=" * 80)
    logger.info("XOR Test Suite - FullIsingModule (1D to 6D)")
    logger.info("=" * 80)
    logger.info(f"Device: {params['device']}")
    logger.info(f"Batch size: {params['batch_size']} | Epochs: {params['epochs']} | Model size: {params['model_size']}")
    logger.info(f"Learning rates: gamma={params['lr_gamma']}, lambda={params['lr_lambda']}, offset={params['lr_offset']}")
    logger.info(f"Annealing: beta={params['sa_beta_range']}, reads={params['num_reads']}, sweeps={params['sa_num_sweeps']}")

    results = {
        'dimensions': [],
        'accuracies': [],
        'losses': [],
        'val_accuracies': [],
        'execution_times': [],
        'all_metrics': [],
    }

    total_start = perf_counter()

    for dim in range(1, 7):
        try:
            dim_start = perf_counter()

            X_train, X_test, y_train, y_test = prepare_xor_data(dim, params)

            accuracy, losses, val_accs, predictions, y_true, probs = train_full_ising_model(
                dim, X_train, y_train, X_test, y_test, params, plotter
            )

            dim_time = perf_counter() - dim_start
            ms = compute_metrics(y_true, probs)

            results['dimensions'].append(dim)
            results['accuracies'].append(accuracy)
            results['losses'].append(losses)
            results['val_accuracies'].append(val_accs)
            results['execution_times'].append(dim_time)
            results['all_metrics'].append(ms)

            logger.info(
                f"\n{dim}D XOR | acc={ms['accuracy']:.4f} prec={ms['precision']:.4f} "
                f"rec={ms['recall']:.4f} f1={ms['f1']:.4f} auc={ms['auc']:.4f} "
                f"| time={dim_time:.2f}s\n"
            )

        except Exception as e:
            logger.error(f"Error on {dim}D XOR: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = perf_counter() - total_start

    if not results['dimensions']:
        logger.error("No valid XOR dimensions processed.")
        return

    # Summary log
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - FullIsingModule XOR Performance Across Dimensions")
    logger.info("=" * 80)
    for dim, acc, t in zip(results['dimensions'], results['accuracies'], results['execution_times']):
        logger.info(f"{dim}D XOR: accuracy={acc:.4f} | time={t:.2f}s")
    logger.info(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    # Summary plots
    dataset_names = [f"{d}D" for d in results['dimensions']]
    plotter.plot_tot_accuracy(results['accuracies'], dataset_names)

    # Metrics CSV
    rows = []
    for i, ds_name in enumerate(dataset_names):
        ms = results['all_metrics'][i]
        row = {'dataset': ds_name, 'model': 'FullIsingModule'}
        for m in METRICS:
            row[f'{m}'] = ms[m]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(str(plotter.output_dir), 'metrics_summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Metrics CSV saved to: {csv_path}")

    # Metrics bar chart (single model, one bar per metric per dataset)
    ms_by_metric = {m: [results['all_metrics'][i][m] for i in range(len(dataset_names))] for m in METRICS}
    zeros = {m: [0.0] * len(dataset_names) for m in METRICS}
    plotter.plot_metrics_bar_comparison(
        ms_by_metric, zeros, dataset_names,
        model_name1='FullIsingModule', model_name2=''
    )

    logger.info(f"\nAll plots saved to {plotter.output_dir}")
    logger.info("FullIsingModule XOR Test Suite Completed!")


if __name__ == '__main__':
    #compare_xor_models_all_dimensions()
    fullIsing_xor_all_dim()

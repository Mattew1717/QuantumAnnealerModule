"""
Test XOR on Network_2L (TwoStageIsingNetwork) from 1D to 6D
Tests XOR problem across multiple dimensions, trains Network_2L, validates, and saves confusion matrices.
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
from time import perf_counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Add repository root to sys.path to enable absolute imports
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.logger import Logger
from Inference.plot import Plot
from Inference.utils import generate_xor_balanced
from Inference.dataset_manager import  HiddenNodesInitialization, SimpleDataset
from ModularNetwork.Network_2L import TwoLayerIsingNetwork
from ModularNetwork.Network_1L import MultiIsingNetwork
from full_ising_model.full_ising_module import FullIsingModule
from full_ising_model.annealers import AnnealingSettings, AnnealerType
from torch.utils.data import DataLoader, TensorDataset

logger = Logger()


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
    SA_settings.sweeps_per_beta = params['sa_sweeps_per_beta']

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
            pred = model(x_batch).view(-1)
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
                preds_tensor = model(test_dataset.x.to(params['device'])).view(-1)
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
        preds_tensor = model(test_dataset.x.to(params['device'])).view(-1)
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
        predictions = np.where(probs < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1']))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

    # Plot loss and accuracy curves
    plotter.plot_loss_accuracy(training_losses, validation_accuracies, f"XOR_{dim}D_Network2L")

    # Plot confusion matrix
    plotter.plot_confusion_matrix(y_test, predictions, labels=[0, 1], filename=f"confusion_matrix_xor_{dim}d")

    return final_accuracy, training_losses, validation_accuracies, predictions, y_test


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
    SA_settings.sweeps_per_beta = params['sa_sweeps_per_beta']

    # Create Network_1L (MultiIsingNetwork)
    model = MultiIsingNetwork(
        num_ising_perceptrons=params['num_ising_net'],  # Use num_ising_1 as the number of perceptrons
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
            pred = model(x_batch).view(-1)
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
                preds_tensor = model(test_dataset.x.to(params['device'])).view(-1)
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
        preds_tensor = model(test_dataset.x.to(params['device'])).view(-1)
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
        predictions = np.where(probs < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1']))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

    # Plot loss and accuracy curves
    plotter.plot_loss_accuracy(training_losses, validation_accuracies, f"XOR_{dim}D_Network1L")

    # Plot confusion matrix
    plotter.plot_confusion_matrix(y_test, predictions, labels=[0, 1], filename=f"confusion_matrix_xor_{dim}d_1L")

    return final_accuracy, training_losses, validation_accuracies, predictions, y_test


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
            accuracy, losses, val_accs, predictions, y_true = train_network_2L(
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
    logger.info(f"Num Ising Perceptrons: {params['num_ising_1']}")
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
            accuracy, losses, val_accs, predictions, y_true = train_network_1L(
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


def test_xor_2d_full_ising_module(params=None, plotter=None):
    """Train and evaluate a FullIsingModule on 2D XOR data."""

    # Start total timer
    total_start = perf_counter()

    logger.info("\n" + "="*60)
    logger.info("Training FullIsingModule on 2D XOR")
    logger.info("="*60)

    # Load params
    if params is None:
        params = get_xor_params()

    # Prepare data (2D)
    X_train, X_test, y_train, y_test = prepare_xor_data(2, params)

    # Determine model size
    if params['model_size'] == -1:
        min_size = int(os.getenv('MINIMUM_MODEL_SIZE', 20))
        model_size = max(X_train.shape[1], min_size)
    else:
        model_size = params['model_size']

    # Create dataloaders
    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, model_size, params
    )

    # Setup annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.sweeps_per_beta = params['sa_sweeps_per_beta']

    # Create FullIsingModule
    model = FullIsingModule(
        size_annealer=model_size,
        annealer_type=AnnealerType.SIMULATED,
        annealing_settings=SA_settings,
        num_reads=params['num_reads'],
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        num_workers=params['num_workers']
    ).to(params['device'])

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

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

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(params['device'])
            y_batch = y_batch.to(params['device']).float()

            optimizer.zero_grad()
            pred = model(x_batch).view(-1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)

        # Validation at each epoch (small datasets)
        model.eval()
        with torch.no_grad():
            preds_tensor = model(test_dataset.x.to(params['device'])).view(-1)
            probs = torch.sigmoid(preds_tensor).cpu().numpy()
            predictions = np.where(probs < 0.5, 0, 1)
            epoch_accuracy = accuracy_score(y_test, predictions)
            validation_accuracies.append(epoch_accuracy)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == params['epochs'] - 1:
            logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {epoch_accuracy:.4f}")

    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation - FullIsingModule on 2D XOR")
    logger.info("="*60)

    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_dataset.x.to(params['device'])).view(-1)
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
        predictions = np.where(probs < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1']))

    cm = confusion_matrix(y_test, predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

    # End total timer
    total_time = perf_counter() - total_start
    logger.info(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    # Plotting (if plotter provided)
    if plotter is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plotter = Plot(output_dir=f'plots_xor_{run_timestamp}')

    plotter.plot_loss_accuracy(training_losses, validation_accuracies, f"XOR_2D_FullIsing")
    plotter.plot_confusion_matrix(y_test, predictions, labels=[0, 1], filename=f"confusion_matrix_xor_2d")

    return final_accuracy, training_losses, validation_accuracies, predictions, y_test


if __name__ == '__main__':
    #test_xor_all_dimensions()
    # Run 2D FullIsingModule quick test
    test_xor_2d_full_ising_module()
    #test_xor_all_dimensions_1L()
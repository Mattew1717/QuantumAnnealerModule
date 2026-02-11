"""
Comparison script between SZP Model (SimAnnealModel) and FullIsingModule.
Tests both models on:
- CSV datasets from Datasets/ directory
- XOR problems from 1D to 6D

All parameters are kept identical between models to demonstrate equivalence.
Uses MSELoss and SGD optimizer for both models.
"""

import os
import sys
import glob
import numpy as np
import torch
import random
from datetime import datetime
from time import perf_counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import dotenv

# Add repository root to sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Load environment variables from Inference directory
_env_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(_env_path)

from Inference.logger import Logger
from Inference.plot import Plot
from Inference.utils import generate_xor_balanced
from Inference.dataset_manager import DatasetManager
from SZP_Model.sim_anneal_model import SimAnnealModel, AnnealingSettings as SZP_AnnealingSettings
from SZP_Model.model import ModelSetting
from SZP_Model.data import SimpleDataset as SZP_SimpleDataset, HiddenNodesInitialization
from SZP_Model.utils import GammaInitialization, utils
from full_ising_model.full_ising_module import FullIsingModule
from full_ising_model.annealers import AnnealingSettings, AnnealerType

logger = Logger()


def get_comparison_params():
    """Load parameters for comparison experiments."""
    seed = int(os.getenv('RANDOM_SEED'))

    # Set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        'random_seed': seed,
        'batch_size': int(os.getenv('BATCH_SIZE')),
        'model_size': int(os.getenv('MODEL_SIZE')),
        'minimum_model_size': int(os.getenv('MINIMUM_MODEL_SIZE')),
        'epochs': int(os.getenv('EPOCHS')),
        'lambda_init': float(os.getenv('LAMBDA_INIT')),
        'offset_init': float(os.getenv('OFFSET_INIT')),
        'learning_rate_gamma': float(os.getenv('LEARNING_RATE_GAMMA')),
        'learning_rate_lambda': float(os.getenv('LEARNING_RATE_LAMBDA')),
        'learning_rate_offset': float(os.getenv('LEARNING_RATE_OFFSET')),
        'sa_beta_range': [int(os.getenv('SA_BETA_MIN')), int(os.getenv('SA_BETA_MAX'))],
        'num_reads': int(os.getenv('NUM_READS')),
        'sa_num_sweeps': int(os.getenv('SA_NUM_SWEEPS')),
        'sa_sweeps_per_beta': int(os.getenv('SA_SWEEPS_PER_BETA')),
        'device': 'cpu',  # Force CPU to avoid CUDA issues
        'num_workers': int(os.getenv('NUM_THREADS', 16)),
        'n_samples_per_region': int(os.getenv('N_SAMPLES_PER_REGION')),
        'test_size': float(os.getenv('TEST_SIZE')),
    }


def prepare_data_for_szp(X_train, y_train, X_test, y_test, model_size):
    """Prepare data for SZP model (SimAnnealModel)."""
    # Standardization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Create SZP datasets
    train_dataset = SZP_SimpleDataset()
    train_dataset.x = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_dataset.y = torch.tensor(y_train, dtype=torch.float32)
    train_dataset.data_size = X_train.shape[1]
    train_dataset.len = len(y_train)

    test_dataset = SZP_SimpleDataset()
    test_dataset.x = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_dataset.y = torch.tensor(y_test, dtype=torch.float32)
    test_dataset.data_size = X_test.shape[1]
    test_dataset.len = len(y_test)

    return train_dataset, test_dataset


def prepare_data_for_pytorch(X_train, y_train, X_test, y_test, model_size):
    """Prepare data for FullIsingModule (PyTorch)."""
    # Standardization (same as SZP)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def train_szp_model(train_dataset, test_dataset, model_size, params):
    """Train SZP Model (SimAnnealModel)."""
    logger.info("Training SZP Model (SimAnnealModel)...")

    # Setup annealing settings
    szp_annealing = SZP_AnnealingSettings()
    szp_annealing.beta_range = params['sa_beta_range']
    szp_annealing.num_reads = params['num_reads']
    szp_annealing.num_sweeps = params['sa_num_sweeps']
    szp_annealing.num_sweeps_per_beta = params['sa_sweeps_per_beta']

    # Create model
    model = SimAnnealModel(size=model_size, settings=szp_annealing)
    model.lmd_init_value = params['lambda_init']
    model.offset_init_value = params['offset_init']

    # Setup model settings
    model.settings = ModelSetting()
    model.settings.gamma_init = GammaInitialization("zeros")  # Common starting point
    model.settings.hidden_nodes_init = HiddenNodesInitialization("function")
    model.settings.hidden_nodes_init.function = SZP_SimpleDataset.offset
    model.settings.hidden_nodes_init.fun_args = [-0.02]
    model.settings.mini_batch_size = params['batch_size']
    model.settings.num_reads = params['num_reads']
    model.settings.optim_steps = params['epochs']
    model.settings.learning_rate_gamma = params['learning_rate_gamma']
    model.settings.learning_rate_lmd = params['learning_rate_lambda']
    model.settings.learning_rate_offset = params['learning_rate_offset']
    model.settings.learning_rate_theta = params['learning_rate_gamma']
    model.settings.dacay_rate = 1.0

    # Train model
    start_time = perf_counter()
    results = model.train(
        training_set=train_dataset,
        test_set=test_dataset,
        verbose=False,
        save_params=False,
        save_samples=False
    )
    training_time = perf_counter() - start_time

    # Extract losses from training
    training_losses = results.results['loss'].tolist()

    # Use test results if available
    if results.results_test is not None and not results.results_test.empty:
        test_losses = results.results_test['loss'].tolist()
    else:
        test_losses = []

    # Compute final predictions and accuracy on test set
    test_dataset_copy = SZP_SimpleDataset()
    test_dataset_copy.x = test_dataset.x.clone()
    test_dataset_copy.y = test_dataset.y.clone()
    test_dataset_copy.data_size = test_dataset.data_size
    test_dataset_copy.len = test_dataset.len

    # Resize if needed
    if test_dataset_copy.data_size < model_size:
        test_dataset_copy.resize(model_size, model.settings.hidden_nodes_init)

    final_predictions = []
    for theta, y_true, _ in test_dataset_copy:
        sample_set = model.eval_single(theta.numpy())
        energy = model._lmd * sample_set.first.energy + model._offset
        final_predictions.append(energy)

    final_predictions = np.array(final_predictions)
    pred_binary = np.where(final_predictions < 0.5, 0, 1)
    y_test_binary = np.where(test_dataset.y.numpy() < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test_binary, pred_binary)
    logger.info(f"  Final Test Accuracy: {final_accuracy:.4f}")

    return training_losses, test_losses, pred_binary, y_test_binary, training_time, final_accuracy


def train_pytorch_model(X_train, y_train, X_test, y_test, model_size, params):
    """Train FullIsingModule (PyTorch)."""
    logger.info("Training FullIsingModule (PyTorch)...")

    # Setup annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.sweeps_per_beta = params['sa_sweeps_per_beta']

    # Create model
    model = FullIsingModule(
        size_annealer=model_size,
        annealer_type=AnnealerType.SIMULATED,
        annealing_settings=SA_settings,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        num_workers=params['num_workers'],
        hidden_nodes_offset_value=-0.02
    ).to(params['device'])

    # Setup optimizer - SGD with separate learning rates for each parameter
    optimizer = torch.optim.SGD([
        {'params': [model.gamma], 'lr': params['learning_rate_gamma']},
        {'params': [model.lmd], 'lr': params['learning_rate_lambda']},
        {'params': [model.offset], 'lr': params['learning_rate_offset']},
    ])
    loss_fn = torch.nn.MSELoss()

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True
    )

    # Training loop
    training_losses = []
    test_losses = []
    validation_accuracies = []
    start_time = perf_counter()

    for epoch in range(params['epochs']):
        model.train()
        epoch_losses = []

        for x_batch, y_batch in train_loader:
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

        # Validation
        model.eval()
        with torch.no_grad():
            preds = model(X_test.to(params['device'])).view(-1).cpu().numpy()
            test_loss = float(loss_fn(torch.tensor(preds), y_test).item())
            test_losses.append(test_loss)

            pred_binary = np.where(preds < 0.5, 0, 1)
            y_test_binary = np.where(y_test.numpy() < 0.5, 0, 1)
            accuracy = accuracy_score(y_test_binary, pred_binary)
            validation_accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f}")

    training_time = perf_counter() - start_time

    # Final predictions and accuracy
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test.to(params['device'])).view(-1).cpu().numpy()
        pred_binary = np.where(final_preds < 0.5, 0, 1)
        y_test_binary = np.where(y_test.numpy() < 0.5, 0, 1)

    final_accuracy = accuracy_score(y_test_binary, pred_binary)
    logger.info(f"  Final Test Accuracy: {final_accuracy:.4f}")

    return training_losses, test_losses, validation_accuracies, pred_binary, y_test_binary, training_time, final_accuracy


def generate_xor_data(dim, params):
    """Generate XOR data for given dimension."""
    logger.info(f"Generating {dim}D XOR data...")

    X, y = generate_xor_balanced(
        dim=dim,
        n_samples_dim=params['n_samples_per_region'],
        shuffle=True,
        random_seed=params['random_seed']
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['test_size'],
        random_state=params['random_seed'],
        stratify=y
    )

    logger.info(f"Total samples: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def compare_on_xor(dim, params, plotter):
    """Compare both models on XOR dataset of given dimension."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing models on {dim}D XOR")
    logger.info(f"{'='*80}")

    try:
        # Generate data
        X_train, X_test, y_train, y_test = generate_xor_data(dim, params)

        # Determine model size
        if params['model_size'] == -1:
            model_size = max(X_train.shape[1], params['minimum_model_size'])
        else:
            model_size = params['model_size']

        logger.info(f"Model size: {model_size}")

        # Train SZP Model
        logger.info("\n--- SZP Model ---")
        train_szp, test_szp = prepare_data_for_szp(X_train, y_train, X_test, y_test, model_size)
        szp_train_losses, szp_test_losses, szp_preds, szp_y_test, szp_time, szp_acc = train_szp_model(
            train_szp, test_szp, model_size, params
        )

        logger.info(f"SZP Model - Final Accuracy: {szp_acc:.4f} | Time: {szp_time:.2f}s")

        # Train PyTorch Model
        logger.info("\n--- FullIsingModule (PyTorch) ---")
        X_train_pt, y_train_pt, X_test_pt, y_test_pt = prepare_data_for_pytorch(
            X_train, y_train, X_test, y_test, model_size
        )
        pt_train_losses, pt_test_losses, pt_val_accs, pt_preds, pt_y_test, pt_time, pt_acc = train_pytorch_model(
            X_train_pt, y_train_pt, X_test_pt, y_test_pt, model_size, params
        )

        logger.info(f"PyTorch Model - Final Accuracy: {pt_acc:.4f} | Time: {pt_time:.2f}s")

        # Plot results
        # For SZP: plot training loss with constant final accuracy
        szp_accs_plot = [szp_acc] * len(szp_train_losses)
        plotter.plot_loss_accuracy(szp_train_losses, szp_accs_plot, f"XOR_{dim}D_SZP")

        # For PyTorch: plot with actual validation accuracies
        plotter.plot_loss_accuracy(pt_train_losses, pt_val_accs, f"XOR_{dim}D_PyTorch")

        plotter.plot_confusion_matrix(szp_y_test, szp_preds, labels=[0, 1],
                                     filename=f"confusion_matrix_xor_{dim}d_szp")
        plotter.plot_confusion_matrix(pt_y_test, pt_preds, labels=[0, 1],
                                     filename=f"confusion_matrix_xor_{dim}d_pytorch")

        return {
            'dimension': dim,
            'szp_accuracy': szp_acc,
            'pytorch_accuracy': pt_acc,
            'szp_time': szp_time,
            'pytorch_time': pt_time,
            'szp_train_losses': szp_train_losses,
            'pytorch_train_losses': pt_train_losses
        }

    except Exception as e:
        logger.error(f"Error on {dim}D XOR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_on_dataset(csv_path, params, plotter):
    """Compare both models on a CSV dataset."""
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing models on dataset: {dataset_name}")
    logger.info(f"{'='*80}")

    try:
        # Load dataset
        dataset_manager = DatasetManager()
        X, y = dataset_manager.load_csv_dataset(csv_path)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params['test_size'],
            random_state=params['random_seed'],
            stratify=y
        )

        logger.info(f"Total samples: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}")

        # Determine model size
        if params['model_size'] == -1:
            model_size = max(X_train.shape[1], params['minimum_model_size'])
        else:
            model_size = params['model_size']

        logger.info(f"Model size: {model_size}")

        # Train SZP Model
        logger.info("\n--- SZP Model ---")
        train_szp, test_szp = prepare_data_for_szp(X_train, y_train, X_test, y_test, model_size)
        szp_train_losses, szp_test_losses, szp_preds, szp_y_test, szp_time, szp_acc = train_szp_model(
            train_szp, test_szp, model_size, params
        )

        logger.info(f"SZP Model - Final Accuracy: {szp_acc:.4f} | Time: {szp_time:.2f}s")

        # Train PyTorch Model
        logger.info("\n--- FullIsingModule (PyTorch) ---")
        X_train_pt, y_train_pt, X_test_pt, y_test_pt = prepare_data_for_pytorch(
            X_train, y_train, X_test, y_test, model_size
        )
        pt_train_losses, pt_test_losses, pt_val_accs, pt_preds, pt_y_test, pt_time, pt_acc = train_pytorch_model(
            X_train_pt, y_train_pt, X_test_pt, y_test_pt, model_size, params
        )

        logger.info(f"PyTorch Model - Final Accuracy: {pt_acc:.4f} | Time: {pt_time:.2f}s")

        # Plot results
        # For SZP: plot training loss with constant final accuracy
        szp_accs_plot = [szp_acc] * len(szp_train_losses)
        plotter.plot_loss_accuracy(szp_train_losses, szp_accs_plot, f"{dataset_name}_SZP")

        # For PyTorch: plot with actual validation accuracies
        plotter.plot_loss_accuracy(pt_train_losses, pt_val_accs, f"{dataset_name}_PyTorch")

        plotter.plot_confusion_matrix(szp_y_test, szp_preds, labels=[0, 1],
                                     filename=f"confusion_matrix_{dataset_name}_szp")
        plotter.plot_confusion_matrix(pt_y_test, pt_preds, labels=[0, 1],
                                     filename=f"confusion_matrix_{dataset_name}_pytorch")

        return {
            'dataset': dataset_name,
            'szp_accuracy': szp_acc,
            'pytorch_accuracy': pt_acc,
            'szp_time': szp_time,
            'pytorch_time': pt_time,
            'szp_train_losses': szp_train_losses,
            'pytorch_train_losses': pt_train_losses
        }

    except Exception as e:
        logger.error(f"Error on dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_full_comparison():
    """Run full comparison: XOR 1D-6D + all CSV datasets."""

    # Initialize
    params = get_comparison_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'comparison_results_{run_timestamp}'

    # Setup plotter and logger
    plotter = Plot(output_dir=output_dir)
    global logger
    logger = Logger(log_dir=output_dir)

    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON: SZP vs FullIsingModule (PyTorch)")
    logger.info("="*80)
    logger.info(f"Random Seed: {params['random_seed']}")
    logger.info(f"Device: {params['device']}")
    logger.info(f"Optimizer: SGD")
    logger.info(f"  - LR Gamma: {params['learning_rate_gamma']}")
    logger.info(f"  - LR Lambda: {params['learning_rate_lambda']}")
    logger.info(f"  - LR Offset: {params['learning_rate_offset']}")
    logger.info(f"Loss Function: MSELoss")
    logger.info(f"Gamma Init: zeros (common starting point)")
    logger.info(f"Batch size: {params['batch_size']}")
    logger.info(f"Epochs: {params['epochs']}")
    logger.info(f"Model size: {params['model_size']}")
    logger.info(f"Annealing: beta={params['sa_beta_range']}, reads={params['num_reads']}, sweeps={params['sa_num_sweeps']}")

    all_results = []

    # # XOR tests (1D to 6D)
    # logger.info("\n" + "="*80)
    # logger.info("TESTING ON XOR DATASETS (1D to 6D)")
    # logger.info("="*80)

    # for dim in range(1, 7):
    #     result = compare_on_xor(dim, params, plotter)
    #     if result:
    #         all_results.append(result)

    # CSV datasets
    logger.info("\n" + "="*80)
    logger.info("TESTING ON CSV DATASETS")
    logger.info("="*80)

    datasets_dir = os.path.join(os.path.dirname(__file__), 'Datasets')
    csv_files = sorted(glob.glob(os.path.join(datasets_dir, '*.csv')))

    if not csv_files:
        logger.warning(f"No CSV files found in {datasets_dir}")

    for csv_path in csv_files:
        result = compare_on_dataset(csv_path, params, plotter)
        if result:
            all_results.append(result)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)

    xor_results = [r for r in all_results if 'dimension' in r]
    dataset_results = [r for r in all_results if 'dataset' in r]

    if xor_results:
        logger.info("\n--- XOR Results ---")
        for r in xor_results:
            logger.info(f"{r['dimension']}D XOR:")
            logger.info(f"  SZP: {r['szp_accuracy']:.4f} ({r['szp_time']:.2f}s)")
            logger.info(f"  PyTorch: {r['pytorch_accuracy']:.4f} ({r['pytorch_time']:.2f}s)")
            logger.info(f"  Difference: {r['pytorch_accuracy'] - r['szp_accuracy']:+.4f}")

    if dataset_results:
        logger.info("\n--- Dataset Results ---")
        for r in dataset_results:
            logger.info(f"{r['dataset']}:")
            logger.info(f"  SZP: {r['szp_accuracy']:.4f} ({r['szp_time']:.2f}s)")
            logger.info(f"  PyTorch: {r['pytorch_accuracy']:.4f} ({r['pytorch_time']:.2f}s)")
            logger.info(f"  Difference: {r['pytorch_accuracy'] - r['szp_accuracy']:+.4f}")

    # Create comparison plots
    if all_results:
        # Accuracy comparison
        labels = []
        szp_accs = []
        pt_accs = []

        for r in all_results:
            if 'dimension' in r:
                labels.append(f"{r['dimension']}D XOR")
            else:
                labels.append(r['dataset'])
            szp_accs.append(r['szp_accuracy'])
            pt_accs.append(r['pytorch_accuracy'])

        plotter.plot_compare_accuracy(
            szp_accs, pt_accs, labels,
            model_name1="SZP Model",
            model_name2="PyTorch FullIsingModule"
        )

        plotter.plot_parity_scatter(
            szp_accs, pt_accs, labels,
            model_name1="SZP Model",
            model_name2="PyTorch FullIsingModule"
        )

        # Overall accuracy bar plots
        plotter.plot_tot_accuracy(szp_accs, labels)

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("\nComparison completed!")


if __name__ == '__main__':
    run_full_comparison()

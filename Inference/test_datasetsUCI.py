import os
import sys
import glob
import time
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import accuracy_score
import dotenv
from scipy.stats import wilcoxon

# Load environment variables
dotenv.load_dotenv()

# Add repository root to sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.logger import Logger
from Inference.dataset_manager import DatasetManager
from Inference.plot import Plot
from full_ising_model.full_ising_module import FullIsingModule
from ModularNetwork.network_1L import MultiIsingNetwork
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
        'sa_num_reads': int(os.getenv('SA_NUM_READS')),
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
    from Inference.dataset_manager import SimpleDataset
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
    SA_settings.sweeps_per_beta = params['sa_sweeps_per_beta']

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
            pred = model(x_batch).view(-1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)
        if epoch % 1 == 0:
                logger.info(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f}")
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            preds_tensor = model(test_set.x.to(params['device'])).view(-1)
            probs = torch.sigmoid(preds_tensor).cpu().numpy()
            predictions = np.where(probs < 0.5, 0, 1)
            epoch_accuracy = accuracy_score(y_test, predictions)
            validation_accuracies.append(epoch_accuracy)

    training_time = time.time() - start_time

    # Final accuracy is the last one
    final_accuracy = validation_accuracies[-1]

    return final_accuracy, training_losses, validation_accuracies


def train_neural_net(X_train, y_train, X_test, y_test, params):
    """Train and evaluate neural Ising network."""

    # Standardization without leakage
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Create datasets without manual resize (FullIsingModule handles it)
    from Inference.dataset_manager import SimpleDataset
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
        from Inference.dataset_manager import HiddenNodesInitialization
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
    SA_settings.sweeps_per_beta = params['sa_sweeps_per_beta']
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
            pred = model(x_batch).view(-1)
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
            preds_tensor = model(test_set.x.to(params['device'])).view(-1)
            probs = torch.sigmoid(preds_tensor).cpu().numpy()
            predictions = np.where(probs < 0.5, 0, 1)
            epoch_accuracy = accuracy_score(test_set.y.numpy(), predictions)
            validation_accuracies.append(epoch_accuracy)

    training_time = time.time() - start_time

    # Final accuracy is the last one
    final_accuracy = validation_accuracies[-1]

    return final_accuracy, training_losses, validation_accuracies


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
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
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

            for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds, 1):
                logger.info(f"Fold {fold_idx}/{params['k_folds']}")

                # Train Single Model
                logger.info("  Training Single Model...")
                acc_s, loss_s, val_accs_s = train_single_model(X_train, y_train, X_test, y_test, params)
                logger.info(f"  Single Model Accuracy: {acc_s:.4f}")

                # Train Neural Network
                logger.info("  Training Neural Network...")
                acc_n, loss_n, val_accs_n = train_neural_net(X_train, y_train, X_test, y_test, params)
                logger.info(f"  Neural Network Accuracy: {acc_n:.4f}")

                single_accs.append(acc_s)
                neural_accs.append(acc_n)
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

            # Calculate mean accuracies for this dataset
            mean_acc_s = np.mean(single_accs)
            mean_acc_n = np.mean(neural_accs)
            std_acc_s = np.std(single_accs)
            std_acc_n = np.std(neural_accs)

            logger.info(f"\nResults | Single: {mean_acc_s:.4f}±{std_acc_s:.4f} | Neural: {mean_acc_n:.4f}±{std_acc_n:.4f} | Diff: {mean_acc_n - mean_acc_s:+.4f}\n")

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

    logger.info(f"All plots saved to {plotter.output_dir}")


if __name__ == '__main__':
    # Import test functions
    from Inference.test_xor import test_xor_all_dimensions, test_xor_all_dimensions_1L

    # Uncomment the test you want to run:

    # Test Network_2L (TwoLayerIsingNetwork) on XOR 1D-6D
    #test_xor_all_dimensions()

    # Test Network_1L (MultiIsingNetwork) on XOR 1D-6D
    #test_xor_all_dimensions_1L()

    # Compare Single vs Multi Ising on all datasets
    # compare_models()

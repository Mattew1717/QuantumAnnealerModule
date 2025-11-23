import os
import glob
import time
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from pathlib import Path
from plot import (
    plot_training_loss,
    plot_heatmap,
    plot_improvement_bars,
    plot_parity_scatter,
    plot_violin,
    plot_dashboard,
)
import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Ensure repository root is on sys.path
import sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from NeuralNetworkIsing.data_ import SimpleDataset, HiddenNodesInitialization
from NeuralNetworkIsing.NeuralNetIsing import MultiIsingNetwork
from TorchIsingModule.IsingModule import FullIsingModule
from IsingModule.ising_learning_model.sim_anneal_model import AnnealingSettings


class Logger:
    """Custom logger that writes to both console and file."""
    
    def __init__(self, log_dir='logs', run_timestamp=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_timestamp = run_timestamp
        
        log_file = self.log_dir / f'run_{run_timestamp}.log'
        
        # Configure logging
        self.logger = logging.getLogger('IsingComparison')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler 
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.info(f"Logging to: {log_file}")
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)


def _getenv_parsed(name, default=None):
    """Parse environment variables with support for Python literals."""
    v = os.getenv(name, None)
    if v is None:
        return default
    try:
        return ast.literal_eval(v)
    except Exception:
        lv = v.strip().lower()
        if lv in ("true", "false"):
            return lv == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except Exception:
            return v


def print_config_header(logger, config, run_timestamp):
    """Print configuration parameters in a professional format."""
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "ISING NEURAL NETWORK - MODEL COMPARISON".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")
    
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ RUN INFORMATION".ljust(79) + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│ Timestamp:  {run_timestamp}".ljust(79) + "│")
    logger.info(f"│ Device:     {config['device']}".ljust(79) + "│")
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")
    
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ DATASET CONFIGURATION".ljust(79) + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│ Classes:           {config['classes']}".ljust(79) + "│")
    logger.info(f"│ Batch Size:        {config['batch_size']}".ljust(79) + "│")
    logger.info(f"│ Random Seed:       {config.get('random_seed', 'N/A')}".ljust(79) + "│")
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")
    
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ MODEL ARCHITECTURE".ljust(79) + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│ Model Size:            {config['size']}".ljust(79) + "│")
    logger.info(f"│ Num Perceptrons:       {config['num_ising_perceptrons']}".ljust(79) + "│")
    logger.info(f"│ Partition Input:       {config['partition_input']}".ljust(79) + "│")
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")
    
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ TRAINING PARAMETERS".ljust(79) + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│ Epochs:                {config['epochs']}".ljust(79) + "│")
    logger.info(f"│ Lambda Init:           {config['lambda_init']:.4f}".ljust(79) + "│")
    logger.info(f"│ Offset Init:           {config['offset_init']:.4f}".ljust(79) + "│")
    logger.info("│".ljust(79) + "│")
    logger.info("│ Learning Rates:".ljust(79) + "│")
    logger.info(f"│   - Gamma:             {config['lr_gamma']:.4f}".ljust(79) + "│")
    logger.info(f"│   - Lambda:            {config['lr_lambda']:.4f}".ljust(79) + "│")
    logger.info(f"│   - Offset:            {config['lr_offset']:.4f}".ljust(79) + "│")
    logger.info(f"│   - Combiner:          {config['lr_combiner']:.4f}".ljust(79) + "│")
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")
    
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ SIMULATED ANNEALING SETTINGS".ljust(79) + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│ Beta Range:            {config['SA_beta_range']}".ljust(79) + "│")
    logger.info(f"│ Num Reads:             {config['SA_num_reads']}".ljust(79) + "│")
    logger.info(f"│ Num Sweeps:            {config['SA_num_sweeps']}".ljust(79) + "│")
    logger.info(f"│ Sweeps per Beta:       {config['SA_sweeps_per_beta']}".ljust(79) + "│")
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")


def load_csv_dataset(csv_path, classes, logger):
    """Load and preprocess CSV dataset."""
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    
    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def prepare_datasets(X_train, y_train, X_val, y_val, config, logger):
    """Prepare datasets with hidden nodes initialization."""
    
    dataset = SimpleDataset()
    test_set = SimpleDataset()
    
    dataset.x = torch.tensor(X_train, dtype=torch.float32)
    dataset.y = torch.tensor(y_train, dtype=torch.float32)
    dataset.data_size = config['input_dim']
    dataset.len = len(y_train)
    
    test_set.x = torch.tensor(X_val, dtype=torch.float32)
    test_set.y = torch.tensor(y_val, dtype=torch.float32)
    test_set.data_size = config['input_dim']
    test_set.len = len(y_val)
    
    # Hidden nodes initialization
    hn = HiddenNodesInitialization(config.get('HN_init', 'function'))
    if config.get('HN_function'):
        hn.function = config.get('HN_function')
    hn.fun_args = config.get('HN_fun_args', None)
    
    # Resize datasets
    if config.get('partition_input'):
        target_size = config['size'] * config['num_ising_perceptrons']
    else:
        target_size = config['size']
    
    logger.info(f"    Resizing datasets to size: {target_size}")
    dataset.resize(target_size, hn)
    test_set.resize(target_size, hn)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(dataset.x, dataset.y),
        batch_size=config['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_set.x, test_set.y),
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    logger.info(f"    Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return dataset, test_set, train_loader, test_loader


def train_single_model(X_train, y_train, X_val, y_val, config, logger):
    """Train and evaluate single Ising model."""
    logger.info("\n  >> Training Single Ising Model")
    
    dataset, test_set, train_loader, test_loader = prepare_datasets(
        X_train, y_train, X_val, y_val, config, logger
    )
    
    # Setup simulated annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = config.get('SA_beta_range', [1, 10])
    SA_settings.num_reads = config.get('SA_num_reads', 1)
    SA_settings.num_sweeps = config.get('SA_num_sweeps', 100)
    SA_settings.sweeps_per_beta = config.get('SA_sweeps_per_beta', 1)
    
    logger.info(f"    Model size: {config['size']}")
    logger.info(f"    SA settings: beta={SA_settings.beta_range}, reads={SA_settings.num_reads}, sweeps={SA_settings.num_sweeps}")
    
    # Create model
    model = FullIsingModule(
        config['size'],
        SA_settings,
        config['lambda_init'],
        config['offset_init']
    ).to(config['device'])
    
    # Setup optimizer
    optimizer = torch.optim.SGD([
        {'params': [model.ising_layer.gamma], 'lr': config['lr_gamma']},
        {'params': [model.lmd], 'lr': config['lr_lambda']},
        {'params': [model.offset], 'lr': config['lr_offset']},
    ])
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    logger.info(f"    Training for {config['epochs']} epochs...")
    training_losses = []
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = []
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(config['device'])
            y_batch = y_batch.to(config['device']).float()
            
            optimizer.zero_grad()
            pred = model(x_batch).view(-1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"      Epoch {epoch+1}/{config['epochs']}: Loss={avg_loss:.4f}, Lambda={model.lmd.item():.4f}, Offset={model.offset.item():.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"    Training completed in {training_time:.2f}s")
    logger.info(f"    Final Lambda: {model.lmd.item():.4f}, Offset: {model.offset.item():.4f}")
    
    # Evaluation
    logger.info("    Evaluating model...")
    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_set.x.to(config['device'])).view(-1)
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
    
    classes = config['classes']
    predictions = np.where(probs < 0.5, classes[0], classes[1])
    accuracy = accuracy_score(y_val, predictions)
    
    # DIAGNOSTIC: Check prediction distribution
    pred_dist = np.bincount(predictions.astype(int))
    logger.info(f"    Prediction distribution: {pred_dist}")
    logger.info(f"    Prob range: [{probs.min():.4f}, {probs.max():.4f}], Mean: {probs.mean():.4f}")
    logger.info(f"    Single Model Accuracy: {accuracy:.4f}")
    
    return accuracy, training_losses


def train_neural_net(X_train, y_train, X_val, y_val, config, logger):
    """Train and evaluate neural Ising network."""
    logger.info("\n  >> Training Neural Ising Network")
    
    dataset = SimpleDataset()
    test_set = SimpleDataset()
    
    dataset.x = torch.tensor(X_train, dtype=torch.float32)
    dataset.y = torch.tensor(y_train, dtype=torch.float32)
    dataset.data_size = config['input_dim']
    dataset.len = len(y_train)
    
    test_set.x = torch.tensor(X_val, dtype=torch.float32)
    test_set.y = torch.tensor(y_val, dtype=torch.float32)
    test_set.data_size = config['input_dim']
    test_set.len = len(y_val)
    
    # Hidden nodes initialization
    hn = HiddenNodesInitialization(config.get('HN_init', 'function'))
    if config.get('HN_function'):
        hn.function = config.get('HN_function')
    hn.fun_args = config.get('HN_fun_args', None)
    
    # Resize datasets
    if config.get('partition_input'):
        target_size = config['size'] * config['num_ising_perceptrons']
    else:
        target_size = config['size']
    
    logger.info(f"    Resizing datasets to size: {target_size}")
    dataset.resize(target_size, hn)
    test_set.resize(target_size, hn)
    
    train_loader = DataLoader(
        TensorDataset(dataset.x, dataset.y),
        batch_size=config['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_set.x, test_set.y),
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    logger.info(f"    Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Setup simulated annealing
    SA_settings = AnnealingSettings()
    SA_settings.beta_range = config.get('SA_beta_range', [1, 10])
    SA_settings.num_reads = config.get('SA_num_reads', 1)
    SA_settings.num_sweeps = config.get('SA_num_sweeps', 100)
    SA_settings.sweeps_per_beta = config.get('SA_sweeps_per_beta', 1)
    
    logger.info(f"    Number of perceptrons: {config['num_ising_perceptrons']}")
    logger.info(f"    Perceptron size: {config['size']}")
    logger.info(f"    Partition input: {config.get('partition_input', False)}")
    
    # Create model
    model = MultiIsingNetwork(
        num_ising_perceptrons=config['num_ising_perceptrons'],
        sizeAnnealModel=config['size'],
        anneal_settings=SA_settings,
        lambda_init=config['lambda_init'],
        offset_init=config['offset_init'],
        partition_input=config.get('partition_input', False),
    ).to(config['device'])
    
    # DIAGNOSTIC: Print initial parameters
    logger.info(f"    Initial combiner weights: {model.combiner_layer.weight.data}")
    logger.info(f"    Initial combiner bias: {model.combiner_layer.bias.data}")
    for i, perceptron in enumerate(model.ising_perceptrons_layer):
        logger.info(f"    Perceptron {i} - Lambda: {perceptron.lmd.item():.4f}, Offset: {perceptron.offset.item():.4f}")
    
    # Setup optimizer
    optimizer_grouped_parameters = []
    for p_idx, single_module in enumerate(model.ising_perceptrons_layer):
        optimizer_grouped_parameters.append({
            'params': [single_module.ising_layer.gamma],
            'lr': config['lr_gamma']
        })
        optimizer_grouped_parameters.append({
            'params': [single_module.lmd],
            'lr': config['lr_lambda']
        })
        optimizer_grouped_parameters.append({
            'params': [single_module.offset],
            'lr': config['lr_offset']
        })
    optimizer_grouped_parameters.append({
        'params': model.combiner_layer.parameters(),
        'lr': config.get('lr_combiner', 0.01)
    })
    
    optimizer = torch.optim.SGD(optimizer_grouped_parameters)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    logger.info(f"    Training for {config['epochs']} epochs...")
    training_losses = []
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = []
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(config['device'])
            y_batch = y_batch.to(config['device']).float()
            
            optimizer.zero_grad()
            pred = model(x_batch).view(-1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # DIAGNOSTIC: Print parameter evolution
            avg_lambda = np.mean([p.lmd.item() for p in model.ising_perceptrons_layer])
            avg_offset = np.mean([p.offset.item() for p in model.ising_perceptrons_layer])
            logger.info(f"      Epoch {epoch+1}/{config['epochs']}: Loss={avg_loss:.4f}, Avg Lambda={avg_lambda:.4f}, Avg Offset={avg_offset:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"    Training completed in {training_time:.2f}s")
    
    # DIAGNOSTIC: Print final parameters
    logger.info(f"    Final combiner weights: {model.combiner_layer.weight.data}")
    logger.info(f"    Final combiner bias: {model.combiner_layer.bias.data}")
    for i, perceptron in enumerate(model.ising_perceptrons_layer):
        logger.info(f"    Perceptron {i} - Lambda: {perceptron.lmd.item():.4f}, Offset: {perceptron.offset.item():.4f}")
    
    # Evaluation
    logger.info("    Evaluating model...")
    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_set.x.to(config['device'])).view(-1)
        probs = torch.sigmoid(preds_tensor).cpu().numpy()
    
    classes = config['classes']
    predictions = np.where(probs < 0.5, classes[0], classes[1])
    accuracy = accuracy_score(test_set.y.numpy(), predictions)
    
    # DIAGNOSTIC: Check prediction distribution
    pred_dist = np.bincount(predictions.astype(int))
    logger.info(f"    Prediction distribution: {pred_dist}")
    logger.info(f"    Prob range: [{probs.min():.4f}, {probs.max():.4f}], Mean: {probs.mean():.4f}")
    logger.info(f"    Neural Network Accuracy: {accuracy:.4f}")
    
    return accuracy, training_losses


def save_plots(dataset_names, accs_single, accs_neural, run_timestamp, logger):
    """Generate and save comparison plots with timestamp."""
    logger.info("\n[4/5] Generating comparison plots...")
    
    plots_dir = Path('Plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Single model accuracies
    logger.info("  Creating plot 1: Single Model accuracies...")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars1 = ax1.bar(range(len(dataset_names)), accs_single, color='#1f77b4', alpha=0.8)
    ax1.set_xticks(range(len(dataset_names)))
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Single Ising Model - Accuracy per Dataset\n[Run: {run_timestamp}]', fontsize=14, pad=15)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accs_single):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.legend()
    fig1.tight_layout()
    plot1_path = plots_dir / f'comparison_single_model_{run_timestamp}.png'
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"    Saved: {plot1_path}")
    
    # Plot 2: Neural network accuracies
    logger.info("  Creating plot 2: Neural Network accuracies...")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bars2 = ax2.bar(range(len(dataset_names)), accs_neural, color='#ff7f0e', alpha=0.8)
    ax2.set_xticks(range(len(dataset_names)))
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Neural Ising Network - Accuracy per Dataset\n[Run: {run_timestamp}]', fontsize=14, pad=15)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accs_neural):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.legend()
    fig2.tight_layout()
    plot2_path = plots_dir / f'comparison_neural_network_{run_timestamp}.png'
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"    Saved: {plot2_path}")
    
    # Plot 3: Side-by-side comparison
    logger.info("  Creating plot 3: Side-by-side comparison...")
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, accs_single, width, label='Single Ising', color='#1f77b4', alpha=0.8)
    bars2 = ax3.bar(x + width/2, accs_neural, width, label='Neural Network', color='#ff7f0e', alpha=0.8)
    
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title(f'Model Comparison: Single vs Neural Network\n[Run: {run_timestamp}]', fontsize=14, pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax3.set_ylim(0, 1.05)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig3.tight_layout()
    plot3_path = plots_dir / f'comparison_sidebyside_{run_timestamp}.png'
    fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    logger.info(f"    Saved: {plot3_path}")
    
    # Plot 4: Boxplot comparison
    logger.info("  Creating plot 4: Boxplot comparison...")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    
    bp = ax4.boxplot(
        [accs_single, accs_neural],
        labels=['Single Ising', 'Neural Network'],
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )
    
    # Color the boxes
    colors = ['#1f77b4', '#ff7f0e']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title(f'Model Comparison Distribution\n[Run: {run_timestamp}]', fontsize=14, pad=15)
    ax4.set_ylim(0, 1.05)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax4.legend()
    
    fig4.tight_layout()
    plot4_path = plots_dir / f'comparison_boxplot_{run_timestamp}.png'
    fig4.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    logger.info(f"    Saved: {plot4_path}")
    
    return [plot1_path, plot2_path, plot3_path, plot4_path]


def compare_models():
    """Compare Single Ising Model vs Neural Ising Network on all datasets."""

    # Initialize timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Logger
    logger = Logger(run_timestamp=run_timestamp)
    logger.info("")

    # Load from environment
    classes = _getenv_parsed('CLASSES', [0, 1])
    batch_size = int(_getenv_parsed('BATCH_SIZE', 32))
    epochs = int(_getenv_parsed('EPOCHS', 100))
    size = int(_getenv_parsed('SIZE', 10))
    num_isings = int(_getenv_parsed('NUM_ISING_PERCEPTRONS', 5))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = int(_getenv_parsed('RANDOM_SEED', 42))

    # Seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Config dict
    config = {
        'classes': classes,
        'batch_size': batch_size,
        'epochs': epochs,
        'size': size,
        'num_ising_perceptrons': num_isings,
        'lambda_init': float(_getenv_parsed('LAMBDA_INIT', 0.1)),
        'offset_init': float(_getenv_parsed('OFFSET_INIT', 0)),
        'lr_gamma': float(_getenv_parsed('LEARNING_RATE_GAMMA', 0.02)),
        'lr_lambda': float(_getenv_parsed('LEARNING_RATE_LAMBDA', 0.01)),
        'lr_offset': float(_getenv_parsed('LEARNING_RATE_OFFSET', 0.01)),
        'lr_combiner': float(_getenv_parsed('LR_COMBINER', 0.1)),
        'device': device,
        'random_seed': random_seed,
        'partition_input': _getenv_parsed('PARTITION_INPUT', False),
        'HN_init': 'function',
        'HN_function': SimpleDataset.offset,
        'HN_fun_args': [-0.01],
        'SA_beta_range': _getenv_parsed('SA_beta_range', [1, 10]),
        'SA_num_reads': int(_getenv_parsed('SA_num_reads', 1)),
        'SA_num_sweeps': int(_getenv_parsed('SA_num_sweeps', 100)),
        'SA_sweeps_per_beta': int(_getenv_parsed('SA_sweeps_per_beta', 1)),
    }

    print_config_header(logger, config, run_timestamp)

    # Dataset discovery
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ DATASET DISCOVERY".ljust(79) + "│")
    logger.info("├" + "─" * 78 + "┤")

    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    csv_files = sorted(glob.glob(os.path.join(datasets_dir, '*.csv')))

    if not csv_files:
        logger.error("│ ERROR: No CSV files found in datasets/ directory!".ljust(79) + "│")
        logger.info("└" + "─" * 78 + "┘")
        return

    logger.info(f"│ Found {len(csv_files)} dataset(s):".ljust(79) + "│")
    for csv in csv_files:
        logger.info(f"│   • {os.path.basename(csv)}".ljust(79) + "│")
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")

    # Training phase
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║ TRAINING PHASE".center(78) + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    dataset_names = []
    accs_single = []
    accs_neural = []

    # NEW: store all training losses
    single_losses_all = []
    neural_losses_all = []

    total_start = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info("┌" + "─" * 78 + "┐")
        logger.info(f"│ Dataset {idx}/{len(csv_files)}: {name}".ljust(79) + "│")
        logger.info("└" + "─" * 78 + "┘")

        try:
            X, y = load_csv_dataset(csv_path, classes, logger)

            split = int(len(y) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            logger.info(f"  Train samples: {len(y_train)}, Test samples: {len(y_val)}")

            input_dim = X_train.shape[1]
            config['input_dim'] = input_dim
            logger.info(f"  Input dimension: {input_dim}")

            acc_s, loss_s = train_single_model(X_train, y_train, X_val, y_val, config, logger)
            acc_n, loss_n = train_neural_net(X_train, y_train, X_val, y_val, config, logger)

            dataset_names.append(name)
            accs_single.append(acc_s)
            accs_neural.append(acc_n)

            single_losses_all.append(loss_s)
            neural_losses_all.append(loss_n)

            logger.info("")
            logger.info("  " + "─" * 76)
            logger.info(f"  RESULTS for {name}:")
            logger.info(f"    Single Model: {acc_s:.4f}")
            logger.info(f"    Neural Net:   {acc_n:.4f}")
            logger.info(f"    Difference:   {acc_n - acc_s:+.4f}")
            logger.info("  " + "─" * 76)
            logger.info("")

        except Exception as e:
            logger.error(f"  ERROR processing {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + f"Training completed in {total_time:.2f}s".center(78) + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    # ============================================================
    # ORIGINAL PLOTS
    # ============================================================
    plot_paths = save_plots(dataset_names, accs_single, accs_neural, run_timestamp, logger)

    # ============================================================
    # NEW ADVANCED PLOTS
    # ============================================================
    plot_training_loss(dataset_names, single_losses_all, neural_losses_all, run_timestamp)
    plot_heatmap(dataset_names, accs_single, accs_neural, run_timestamp)
    plot_improvement_bars(dataset_names, accs_single, accs_neural, run_timestamp)
    plot_parity_scatter(dataset_names, accs_single, accs_neural, run_timestamp)
    plot_violin(accs_single, accs_neural, run_timestamp)
    plot_dashboard(dataset_names, accs_single, accs_neural, run_timestamp)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    logger.info("\n[5/5] Saving results...")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    df = pd.DataFrame({
        "Dataset": dataset_names,
        "Single_Accuracy": accs_single,
        "Neural_Accuracy": accs_neural,
        "Difference": [n - s for n, s in zip(accs_neural, accs_single)]
    })

    out_csv = out_dir / f"comparison_results_{run_timestamp}.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"  Saved results to: {out_csv}")

    logger.info("")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║ RUN COMPLETE!".center(78) + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    logger.info("Output files:")
    logger.info(f"  Logs:    logs/run_{run_timestamp}.log")
    logger.info(f"  Results: {out_csv}")

    logger.info(f"  Plots:")
    for p in plot_paths:
        logger.info(f"    • {p}")
    logger.info(f"    • Advanced plots also stored in Plots/")

    logger.info("")
    logger.info("=" * 80)



if __name__ == '__main__':
    compare_models()
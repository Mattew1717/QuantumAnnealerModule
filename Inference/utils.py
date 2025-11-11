import os
import glob
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import dotenv
dotenv.load_dotenv()

# Ensure repository root is on sys.path so sibling packages (NeuralNetworkIsing, TorchIsingModule, IsingModule)
# can be imported when running scripts from the `Inference/` folder.
import sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from NeuralNetworkIsing.data_ import SimpleDataset, HiddenNodesInitialization


# -- Environment helpers -------------------------------------------------
def _get_env(name, default=None, parse_json=False):
    """Get environment variable and optionally parse it as Python literal (list, dict, number).
    """
    val = os.getenv(name)
    if val is None:
        return default
    if parse_json:
        try:
            return ast.literal_eval(val)
        except Exception:
            return default
    return val

def prepare_datasets(X_train, y_train, X_val, y_val, input_dim):
    """Prepare SimpleDataset wrappers and DataLoader instances.
    """
    dataset = SimpleDataset()
    test_set = SimpleDataset()
    dataset.x = torch.tensor(X_train, dtype=torch.float32)
    dataset.y = torch.tensor(y_train, dtype=torch.float32)
    dataset.data_size = input_dim
    dataset.len = len(y_train)

    test_set.x = torch.tensor(X_val, dtype=torch.float32)
    test_set.y = torch.tensor(y_val, dtype=torch.float32)
    test_set.data_size = input_dim
    test_set.len = len(y_val)

    # Read hidden nodes init mode from env, fallback to 'function'
    HN_init = _get_env('HN_init', default='function')
    hn = HiddenNodesInitialization(HN_init)

    # HN_function is usually a callable set in code; allow an override via
    # a provided attribute on the SimpleDataset or via env name (not deserialized)
    # Default: no change
    # If HN_function is a string name of a callable in this module, try to resolve it
    hn_fun_env = _get_env('HN_function', default=None)
    if hn_fun_env is not None and hasattr(SimpleDataset, hn_fun_env):
        hn.function = getattr(SimpleDataset, hn_fun_env)

    # Additional args for hidden nodes initialization
    hn.fun_args = _get_env('HN_fun_args', default=None, parse_json=True)

    SIZE = input_dim + 5  # keep original adjustment

    PARTITION_INPUT = _get_env('PARTITION_INPUT', default=False, parse_json=True)
    NUM_ISING_PERCEPTRONS = _get_env('NUM_ISING_PERCEPTRONS', default=1, parse_json=True)

    if PARTITION_INPUT:
        dataset.resize(SIZE * int(NUM_ISING_PERCEPTRONS), hn)
        dataset.len = len(dataset.y)
        dataset.data_size = len(dataset.x[0])
        test_set.resize(SIZE * int(NUM_ISING_PERCEPTRONS), hn)
        test_set.len = len(test_set.y)
        test_set.data_size = len(test_set.x[0])
    else:
        dataset.resize(SIZE, hn)
        dataset.len = len(dataset.y)
        dataset.data_size = len(dataset.x[0])
        test_set.resize(SIZE, hn)
        test_set.len = len(test_set.y)
        test_set.data_size = len(test_set.x[0])

    BATCH_SIZE = _get_env('BATCH_SIZE', default=32, parse_json=True)

    train_loader = DataLoader(
        TensorDataset(dataset.x, dataset.y),
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_set.x, test_set.y),
        batch_size=int(BATCH_SIZE),
        shuffle=False,
    )

    return dataset, test_set, train_loader, test_loader

# === Plot ===
def plot_function_predictions(x_values, predictions, func=None, ranges=None, save_path=None):
    """Plot predictions against a target function.
    """
    # Resolve function
    if func is None:
        # Do not try to eval callables from env; caller should pass a callable.
        raise ValueError('func must be a callable')

    if ranges is None:
        ranges = _get_env('RANGES_TEST', default=[[0, 1]], parse_json=True)
    x = np.linspace(ranges[0][0], ranges[0][1], 200)
    y = func(x)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label='target', color='blue')
    ax.scatter(x_values, predictions, color='red', label='predictions')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Predictions vs target')
    ax.grid(True)
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_training_loss(training_losses, dataset_name, save_path=None):
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(training_losses) + 1), training_losses, marker='o', color='black', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14, fontname='serif')
    ax.set_ylabel('Training Loss', fontsize=14, fontname='serif')
    ax.set_title('Training Loss Curve', fontsize=16, fontname='serif', pad=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    if save_path is None:
        save_path = get_next_plot_filename('training_loss_NET', dataset_name, 'png')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """Create and save a confusion matrix heatmap.

    labels: list of class labels. If None, attempts to read CLASSES from env.
    """
    plt.style.use('classic')
    if labels is None:
        labels = _get_env('CLASSES', default=[0, 1], parse_json=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greys', cbar=False, ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'}, linewidths=0.5, linecolor='black'
    )
    ax.set_xlabel('Predicted label', fontsize=14, fontname='serif')
    ax.set_ylabel('True label', fontsize=14, fontname='serif')
    ax.set_xticklabels(labels, fontsize=12, fontname='serif')
    ax.set_yticklabels(labels, fontsize=12, fontname='serif', rotation=0)
    ax.set_title('Confusion Matrix', fontsize=16, fontname='serif', pad=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close(fig)
    else:
        plt.show()

def fquad(x):
    return -(x - 0.5) ** 2


def flin(x):
    return 2 * x - 6


def fcub(x):
    return (x - 0.5) ** 3


def flog(x):
    return np.log(x)

def generate_xor_balanced(dim, n_samples_dim=1000, shuffle=True, random_seed=42):
    """Generate XOR data in U[0,1]^d with balanced classes."""
    if random_seed is not None:
        np.random.seed(random_seed)
    samples = np.random.random(size=(2 ** dim * n_samples_dim, dim))
    for i in range(2 ** dim):
        signs = np.array([1 if int((i // 2 ** d) % 2) == 0 else -1 for d in range(dim)])
        samples[i * n_samples_dim:(i + 1) * n_samples_dim] *= signs
    labels = np.sign(np.prod(samples, axis=1))
    if shuffle:
        perm = np.random.permutation(2 ** dim * n_samples_dim)
        samples = samples[perm]
        labels = labels[perm]
    labels = np.where(labels < 0, 0, 1)
    return samples, labels

def get_next_plot_filename(base_name, dataset_name, ext):
    """Return a unique filename under the Plots/ directory."""
    os.makedirs('Plots', exist_ok=True)
    safe_dataset = os.path.splitext(os.path.basename(dataset_name))[0].replace(' ', '_')
    pattern = f'Plots/{base_name}_{safe_dataset}_*.{ext}'
    existing = glob.glob(pattern)
    nums = []
    for f in existing:
        try:
            num = int(os.path.splitext(f)[0].split('_')[-1])
            nums.append(num)
        except ValueError:
            continue
    next_num = max(nums) + 1 if nums else 1
    return f'Plots/{base_name}_{safe_dataset}_{next_num}.{ext}'

def plot_confusion_matrix_scientific(y_true, y_pred, save_path='confusion_matrix_NET.png'):
    """Compatibility wrapper for older call sites."""
    labels = _get_env('CLASSES', default=[0, 1], parse_json=True)
    return plot_confusion_matrix(y_true, y_pred, labels=labels, save_path=save_path)

def plot_results_table_scientific(params, accuracy, training_loss, test_loss, errors_class_0, errors_class_1, training_time, save_path='results_table_NET.png'):
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(8, len(params) * 0.35 + 2))
    ax.axis('off')
    labels = _get_env('CLASSES', default=[0, 1], parse_json=True)
    table_data = params + [
        ['Accuracy', f'{accuracy:.4f}'],
        ['Training Loss', f'{training_loss:.4f}'],
        ['Test Loss', f'{test_loss:.4f}'],
        [f'Errors class {labels[0]}', errors_class_0],
        [f'Errors class {labels[1]}', errors_class_1],
        ['Training time (s)', f'{training_time:.2f}'],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=['Parameter', 'Value'],
        cellLoc='left',
        loc='center',
        edges='horizontal',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        text = cell.get_text()
        text.set_fontname('serif')
        if row == 0:
            cell.set_facecolor('#f5f5f5')
            text.set_fontweight('bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

# alias kept for compatibility: plot_training_loss already defined above


def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    labels = _get_env('CLASSES', default=[0, 1], parse_json=True)
    # replace -1/1 with configured class labels when numeric class mapping is used
    df.iloc[:, -1] = df.iloc[:, -1].replace({-1: labels[0], 1: labels[1]})
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y
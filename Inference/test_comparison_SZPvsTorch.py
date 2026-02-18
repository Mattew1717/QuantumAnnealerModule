"""
Comparison: SZP Model vs FullIsingModule on iris dataset with K-Fold cross-validation.
"""

import os
import sys
import numpy as np
import torch
from time import perf_counter
from sklearn.metrics import accuracy_score
import dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_env_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(_env_path)

from Inference.logger import Logger
from Inference.dataset_manager import DatasetManager
from Inference.plot import Plot
from SZP_Model.sim_anneal_model import SimAnnealModel, AnnealingSettings as SZP_AnnealingSettings
from SZP_Model.data import SimpleDataset as SZP_SimpleDataset, HiddenNodesInitialization
from SZP_Model.utils import GammaInitialization
from full_ising_model.full_ising_module import FullIsingModule
from full_ising_model.annealers import AnnealingSettings, AnnealerType

logger = Logger()


def print_params_table(params):
    """Print a formatted table of all comparison parameters."""
    rows = [
        ('random_seed',          params['random_seed']),
        ('k_folds',              params['k_folds']),
        ('epochs',               params['epochs']),
        ('batch_size',           params['batch_size']),
        ('model_size',           'n_features + 5'),
        ('lambda_init',          params['lambda_init']),
        ('offset_init',          params['offset_init']),
        ('learning_rate_gamma',  params['learning_rate_gamma']),
        ('learning_rate_lambda', params['learning_rate_lambda']),
        ('learning_rate_offset', params['learning_rate_offset']),
        ('sa_beta_range',        params['sa_beta_range']),
        ('num_reads',            params['num_reads']),
        ('sa_num_sweeps',        params['sa_num_sweeps']),
        ('sa_sweeps_per_beta',   params['sa_sweeps_per_beta']),
        ('num_workers',          params['num_workers']),
    ]
    col_w = max(len(r[0]) for r in rows) + 2
    val_w = max(len(str(r[1])) for r in rows) + 2
    sep = '+' + '-' * col_w + '+' + '-' * val_w + '+'
    header = f"| {'Parameter':<{col_w-1}}| {'Value':<{val_w-1}}|"
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for name, val in rows:
        logger.info(f"| {name:<{col_w-1}}| {str(val):<{val_w-1}}|")
    logger.info(sep)


def get_params():
    seed = int(os.getenv('RANDOM_SEED'))
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        'num_workers': int(os.getenv('NUM_THREADS', 16)),
        'k_folds': int(os.getenv('K_FOLDS', 5)),
    }


def get_model_size(n_features, params):
    # if params['model_size'] == -1:
    #     return max(n_features, params['minimum_model_size'])
    return n_features + 5


def train_szp(X_train, y_train, X_test, y_test, model_size, params):
    train_ds = SZP_SimpleDataset()
    train_ds.x = torch.tensor(X_train, dtype=torch.float32)
    train_ds.y = torch.tensor(y_train, dtype=torch.float32)
    train_ds.data_size = X_train.shape[1]
    train_ds.len = len(y_train)

    test_ds = SZP_SimpleDataset()
    test_ds.x = torch.tensor(X_test, dtype=torch.float32)
    test_ds.y = torch.tensor(y_test, dtype=torch.float32)
    test_ds.data_size = X_test.shape[1]
    test_ds.len = len(y_test)

    szp_annealing = SZP_AnnealingSettings()
    szp_annealing.beta_range = params['sa_beta_range']
    szp_annealing.num_reads = params['num_reads']
    szp_annealing.num_sweeps = params['sa_num_sweeps']
    szp_annealing.sweeps_per_beta = params['sa_sweeps_per_beta']

    model = SimAnnealModel(size=model_size, settings=szp_annealing)
    model.lmd_init_value = params['lambda_init']
    model.offset_init_value = params['offset_init']
    model.settings.gamma_init = GammaInitialization("zeros")
    model.settings.hidden_nodes_init = HiddenNodesInitialization("function")
    model.settings.hidden_nodes_init.function = SZP_SimpleDataset.offset
    model.settings.hidden_nodes_init.fun_args = [-0.1]
    model.settings.mini_batch_size = params['batch_size']
    model.settings.num_reads = params['num_reads']
    model.settings.optim_steps = params['epochs']
    model.settings.learning_rate_gamma = params['learning_rate_gamma']
    model.settings.learning_rate_lmd = params['learning_rate_lambda']
    model.settings.learning_rate_offset = params['learning_rate_offset']
    model.settings.learning_rate_theta = params['learning_rate_gamma']
    model.settings.dacay_rate = 1

    t0 = perf_counter()
    model.train(training_set=train_ds, test_set=test_ds, verbose=False, save_params=False, save_samples=False)
    elapsed = perf_counter() - t0

    # Predict on test set
    test_copy = SZP_SimpleDataset()
    test_copy.x = test_ds.x.clone()
    test_copy.y = test_ds.y.clone()
    test_copy.data_size = test_ds.data_size
    test_copy.len = test_ds.len
    if test_copy.data_size < model_size:
        test_copy.resize(model_size, model.settings.hidden_nodes_init)

    preds = []
    for theta, y_true, _ in test_copy:
        sample_set = model.eval_single(theta.numpy())
        energy = model._lmd * sample_set.first.energy + model._offset
        preds.append(energy)

    preds = np.array(preds)
    pred_binary = (preds >= 0.5).astype(int)
    y_binary = (test_ds.y.numpy() >= 0.5).astype(int)
    acc = accuracy_score(y_binary, pred_binary)
    return acc, elapsed


def train_pytorch(X_train, y_train, X_test, y_test, model_size, params):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    SA_settings = AnnealingSettings()
    SA_settings.beta_range = params['sa_beta_range']
    SA_settings.num_reads = params['num_reads']
    SA_settings.num_sweeps = params['sa_num_sweeps']
    SA_settings.sweeps_per_beta = params['sa_sweeps_per_beta']

    model = FullIsingModule(
        size_annealer=model_size,
        annealer_type=AnnealerType.SIMULATED,
        annealing_settings=SA_settings,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        num_workers=params['num_workers'],
        hidden_nodes_offset_value=-0.1,
        gamma_init=torch.zeros((model_size, model_size), dtype=torch.float32)
    )

    optimizer = torch.optim.SGD([
        {'params': [model.gamma], 'lr': params['learning_rate_gamma']},
        {'params': [model.lmd], 'lr': params['learning_rate_lambda']},
        {'params': [model.offset], 'lr': params['learning_rate_offset']},
    ])
    loss_fn = torch.nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=params['batch_size'], shuffle=True
    )

    t0 = perf_counter()
    for epoch in range(params['epochs']):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(x_b).view(-1)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
    elapsed = perf_counter() - t0

    model.eval()
    with torch.no_grad():
        preds = model(X_te).view(-1).cpu().numpy()
    pred_binary = (preds >= 0.5).astype(int)
    y_binary = (y_te.numpy() >= 0.5).astype(int)
    acc = accuracy_score(y_binary, pred_binary)
    return acc, elapsed


def run_comparison():
    params = get_params()
    k = params['k_folds']

    iris_path = os.path.join(os.path.dirname(__file__), 'Datasets', '00_iris_versicolor_virginica.csv')
    dm = DatasetManager()
    X, y = dm.load_csv_dataset(iris_path)
    folds = dm.generate_k_folds(X, y, k)
    model_size = get_model_size(X.shape[1], params)

    logger.info(f"\n{'='*60}")
    logger.info(f"K-Fold ({k}) comparison on iris | model_size={model_size} | epochs={params['epochs']}")
    logger.info(f"{'='*60}")
    print_params_table(params)

    szp_accs, pt_accs = [], []
    szp_times, pt_times = [], []

    for i, (X_train, y_train, X_test, y_test) in enumerate(folds):
        logger.info(f"\n--- Fold {i+1}/{k} ---")

        acc_szp, t_szp = train_szp(X_train, y_train, X_test, y_test, model_size, params)
        logger.info(f"  SZP:     acc={acc_szp:.4f}  time={t_szp:.2f}s")
        szp_accs.append(acc_szp)
        szp_times.append(t_szp)

        acc_pt, t_pt = train_pytorch(X_train, y_train, X_test, y_test, model_size, params)
        logger.info(f"  PyTorch: acc={acc_pt:.4f}  time={t_pt:.2f}s")
        pt_accs.append(acc_pt)
        pt_times.append(t_pt)

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"  SZP     mean acc={np.mean(szp_accs):.4f} ± {np.std(szp_accs):.4f}  avg time={np.mean(szp_times):.2f}s")
    logger.info(f"  PyTorch mean acc={np.mean(pt_accs):.4f} ± {np.std(pt_accs):.4f}  avg time={np.mean(pt_times):.2f}s")
    logger.info(f"{'='*60}")

    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    plotter = Plot(output_dir=plots_dir)
    out = plotter.plot_comparison_accuracies(szp_accs, pt_accs)
    logger.info(f"Accuracy plot saved to: {out}")


if __name__ == '__main__':
    run_comparison()

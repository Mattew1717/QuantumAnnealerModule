
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import dotenv

dotenv.load_dotenv(dotenv_path=Path(__file__).parent / '.env')

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.utils.logger import Logger
from Inference.utils.utils import (
    METRICS,
    build_annealing_settings,
    compute_metrics,
    flatten_logits,
    generate_xor_balanced,
    resolve_model_size,
    set_global_seed,
    standardize_train_test,
)
from full_ising_model.annealers import AnnealerType
from full_ising_model.full_ising_module import FullIsingModule
from NeuralNetwork.modular_network import ModularNetwork

logger = Logger()

DIM = 2                              # 2D XOR
NUM_NODES = 3                        # "rete a 3 nodi"
# Annealer backend from .env: simulated for tuning/statistics, quantum for a
# single QPU run (needs DWAVE_PROFILE and NUM_READS as well).
ANNEALER = AnnealerType(os.environ['ANNEALER_TYPE'])


def get_params():
    """Load the 2D-XOR hyperparameters strictly from .env (no defaults)."""
    return {
        'random_seed': int(os.environ['RANDOM_SEED']),
        'batch_size': int(os.environ['BATCH_SIZE']),
        'model_size': int(os.environ['MODEL_SIZE']),
        'minimum_model_size': int(os.environ['MINIMUM_MODEL_SIZE']),
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
        'n_samples_per_region': int(os.environ['N_SAMPLES_PER_REGION']),
        'test_size': float(os.environ['TEST_SIZE']),
        'partition_input': os.environ['PARTITION_INPUT'].lower() == 'true',
        'profile': os.environ['DWAVE_PROFILE'],  # used only when ANNEALER=quantum
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


# --------------------------------------------------------------------------- #
# Building blocks (shared by the statistics run and the single reference run)  #
# --------------------------------------------------------------------------- #

def _prepare_2d_xor(seed, params):
    """Generate, split and standardize a balanced 2D-XOR dataset for one seed."""
    X, y = generate_xor_balanced(DIM, params['n_samples_per_region'],
                                 shuffle=True, random_seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=params['test_size'], random_state=seed, stratify=y,
    )
    X_tr, X_te = standardize_train_test(X_tr, X_te)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=params['batch_size'], shuffle=True,
    )
    return train_loader, torch.tensor(X_te, dtype=torch.float32), y_te


def _build_full(model_size, params):
    model = FullIsingModule(
        size_annealer=model_size,
        annealer_type=ANNEALER,
        annealing_settings=build_annealing_settings(params),
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        num_workers=params['num_workers'],
        hidden_nodes_offset_value=params['hidden_nodes_offset_value'],
        profile=params['profile'],          # ignored unless ANNEALER=quantum
        num_reads=params['num_reads'],       # ignored unless ANNEALER=quantum
    ).to(params['device'])
    optimizer = torch.optim.Adam([
        {'params': [model.gamma], 'lr': params['lr_gamma']},
        {'params': [model.lmd], 'lr': params['lr_lambda']},
        {'params': [model.offset], 'lr': params['lr_offset']},
    ])
    return model, optimizer


def _build_modular(model_size, params, num_nodes):
    model = ModularNetwork(
        num_ising_perceptrons=num_nodes,
        size_annealer=model_size,
        annealing_settings=build_annealing_settings(params),
        annealer_type=ANNEALER,
        lambda_init=params['lambda_init'],
        offset_init=params['offset_init'],
        hidden_nodes_offset_value=params['hidden_nodes_offset_value'],
        num_workers=params['num_workers'],
        combiner_bias=True,
        partition_input=params['partition_input'],
        profile=params['profile'],          # ignored unless ANNEALER=quantum
        num_reads=params['num_reads'],       # ignored unless ANNEALER=quantum
    ).to(params['device'])
    groups = []
    for module in model.ising_perceptrons_layer:
        groups.append({'params': [module.gamma], 'lr': params['lr_gamma']})
        groups.append({'params': [module.lmd], 'lr': params['lr_lambda']})
        groups.append({'params': [module.offset], 'lr': params['lr_offset']})
    groups.append({'params': model.combiner_layer.parameters(), 'lr': params['lr_combiner']})
    return model, torch.optim.Adam(groups)


def _train(model, optimizer, train_loader, params):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(params['epochs']):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(params['device'])
            y_batch = y_batch.to(params['device']).float()
            optimizer.zero_grad()
            loss = loss_fn(flatten_logits(model(x_batch)), y_batch)
            loss.backward()
            optimizer.step()


def _evaluate(model, X_te, y_te, params):
    model.eval()
    with torch.no_grad():
        logits = flatten_logits(model(X_te.to(params['device'])))
        probs = torch.sigmoid(logits).cpu().numpy()
    return compute_metrics(y_te, probs)


def _run_both_models(seed, params, num_nodes):
    """Train FullIsingModule and an N-node ModularNetwork on one 2D-XOR seed."""
    set_global_seed(seed)
    train_loader, X_te, y_te = _prepare_2d_xor(seed, params)
    model_size = resolve_model_size(DIM, params)

    model_full, opt_full = _build_full(model_size, params)
    _train(model_full, opt_full, train_loader, params)
    metrics_full = _evaluate(model_full, X_te, y_te, params)

    model_mod, opt_mod = _build_modular(model_size, params, num_nodes)
    _train(model_mod, opt_mod, train_loader, params)
    metrics_mod = _evaluate(model_mod, X_te, y_te, params)

    return metrics_full, metrics_mod, model_size


def _log_config(params, num_nodes, header):
    logger.info("\n" + "=" * 80)
    logger.info(header)
    logger.info("=" * 80)
    logger.info(f"Annealer type: {ANNEALER.value}")
    logger.info("Parameter configuration:")
    for key in sorted(params):
        logger.info(f"  {key:<28} = {params[key]}")


def _log_metrics(name, m):
    logger.info(f"{name:<20} | acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
                f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")


# --------------------------------------------------------------------------- #
# Part 1 - multi-run statistics (mean +/- std over N simulated-annealing runs) #
# --------------------------------------------------------------------------- #

def run_sim_statistics(n_runs=5, num_nodes=NUM_NODES):
    """Run the 2D-XOR experiment over `n_runs` seeds and report mean +/- std."""
    if ANNEALER != AnnealerType.SIMULATED:
        raise RuntimeError(
            f"run_sim_statistics issues {n_runs}x the annealer calls and is meant "
            f"for simulated annealing; ANNEALER_TYPE={ANNEALER.value}. Use run_xor_2d() "
            f"for a single run on the QPU."
        )
    params = get_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'sim_stats_xor2d_{run_timestamp}'

    global logger
    logger = Logger(log_dir=out_dir)

    _log_config(params, num_nodes,
                f"2D XOR statistics ({n_runs} runs, simulated annealing) - "
                f"FullIsingModule vs {num_nodes}-node ModularNetwork")

    full_runs = {m: [] for m in METRICS}
    mod_runs = {m: [] for m in METRICS}
    model_size = None

    logger.info(f"\nRunning {n_runs} runs (seeds 0..{n_runs - 1})...")
    for seed in range(n_runs):
        metrics_full, metrics_mod, model_size = _run_both_models(seed, params, num_nodes)
        for m in METRICS:
            full_runs[m].append(metrics_full[m])
            mod_runs[m].append(metrics_mod[m])
        logger.info(
            f"  [run {seed + 1}/{n_runs} | seed {seed}] "
            f"FullIsing acc={metrics_full['accuracy']:.3f} | "
            f"Modular({num_nodes}) acc={metrics_mod['accuracy']:.3f}"
        )

    logger.info(f"\nModel size (annealer spins): {model_size}")
    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS over {n_runs} runs (mean +/- std)")
    logger.info("=" * 80)
    _log_stats('FullIsingModule', full_runs)
    _log_stats(f'ModularNetwork({num_nodes})', mod_runs)
    logger.info(f"\nLog saved to {out_dir}")
    return full_runs, mod_runs


def _log_stats(name, runs):
    logger.info(name)
    for m in METRICS:
        vals = np.array(runs[m], dtype=float)
        logger.info(
            f"  {m:<10} mean={np.nanmean(vals):.4f}  std={np.nanstd(vals):.4f}  "
            f"min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}"
        )


# --------------------------------------------------------------------------- #
# Part 2 - the actual single 2D-XOR run                                        #
# --------------------------------------------------------------------------- #

def run_xor_2d(num_nodes=NUM_NODES):
    """Single 2D-XOR run: log the parameter configuration, train both models.

    Trains FullIsingModule and a `num_nodes`-node ModularNetwork on simulated
    annealing and returns the metrics dict of each model.
    """
    params = get_params()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'xor2d_{run_timestamp}'

    global logger
    logger = Logger(log_dir=out_dir)

    _log_config(params, num_nodes,
                f"2D XOR (simulated annealing) - "
                f"FullIsingModule vs {num_nodes}-node ModularNetwork")

    metrics_full, metrics_mod, model_size = _run_both_models(
        params['random_seed'], params, num_nodes
    )

    logger.info(f"\nModel size (annealer spins): {model_size}")
    _log_metrics('FullIsingModule', metrics_full)
    _log_metrics(f'ModularNetwork({num_nodes})', metrics_mod)
    logger.info(f"Diff(Modular-Single) acc: "
                f"{metrics_mod['accuracy'] - metrics_full['accuracy']:+.4f}")
    logger.info(f"\nLog saved to {out_dir}")
    return metrics_full, metrics_mod


if __name__ == '__main__':
    # simulated -> 5-run statistics; quantum -> a single run of both models.
    if ANNEALER == AnnealerType.SIMULATED:
        run_sim_statistics(n_runs=5)
    else:
        run_xor_2d()

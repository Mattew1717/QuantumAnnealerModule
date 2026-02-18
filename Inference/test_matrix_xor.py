import os
import sys
import traceback
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from time import perf_counter
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import dotenv

dotenv.load_dotenv(dotenv_path=Path(__file__).parent / '.env')

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.logger import Logger
from Inference.utils import generate_xor_balanced
from ModularNetwork.Network_1L import MultiIsingNetwork
from full_ising_model.annealers import AnnealingSettings, AnnealerType
from torch.utils.data import DataLoader, TensorDataset

# ─── grid configuration ───────────────────────────────────────────────────────
NUM_NODI_LIST    = [2, 3, 5, 8, 10, 15]   # num_ising_perceptrons
MULTIPLIERS      = [1, 2, 3, 4]            # proportional node sizes (× num_features)
FIXED_SIZES      = [10, 20, 30, 40, 50]    # fixed node sizes

# ─── metric thresholds ────────────────────────────────────────────────────────
CONV_THRESHOLD   = 0.85   # val accuracy considered "converged"
STABILITY_WINDOW = 10     # last N epochs for loss-stability std

# ─── global run directory (shared across all 6 calls in main) ─────────────────
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_OUTPUT_ROOT   = Path(f"plots_matrix_xor_{_RUN_TIMESTAMP}")
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_GLOBAL_LOGGER = Logger(log_dir=str(_OUTPUT_ROOT))


# ─── helpers ──────────────────────────────────────────────────────────────────

def _get_params() -> dict:
    """Read experiment parameters from environment variables."""
    return {
        'random_seed':        int(os.getenv('RANDOM_SEED', 42)),
        'batch_size':         int(os.getenv('BATCH_SIZE', 256)),
        'epochs':             int(os.getenv('EPOCHS', 200)),
        'lambda_init':        float(os.getenv('LAMBDA_INIT', -0.1)),
        'offset_init':        float(os.getenv('OFFSET_INIT', 0.0)),
        'lr_gamma':           float(os.getenv('LEARNING_RATE_GAMMA', 1e-4)),
        'lr_lambda':          float(os.getenv('LEARNING_RATE_LAMBDA', 1e-4)),
        'lr_offset':          float(os.getenv('LEARNING_RATE_OFFSET', 1e-4)),
        'lr_classical':       float(os.getenv('LEARNING_RATE_COMBINER', 1e-3)),
        'sa_beta_range':      [int(os.getenv('SA_BETA_MIN', 1)),
                               int(os.getenv('SA_BETA_MAX', 10))],
        'num_reads':          int(os.getenv('NUM_READS', 100)),
        'sa_num_sweeps':      int(os.getenv('SA_NUM_SWEEPS', 1000)),
        'sa_sweeps_per_beta': int(os.getenv('SA_SWEEPS_PER_BETA', 1)),
        'n_samples_per_region': int(os.getenv('N_SAMPLES_PER_REGION', 200)),
        'test_size':          float(os.getenv('TEST_SIZE', 0.2)),
        'val_interval':       int(os.getenv('VALIDATION_INTERVAL', 5)),
        'device':             'cuda' if torch.cuda.is_available() else 'cpu',
    }


def _node_size_configs(num_features: int) -> list[tuple[int, str]]:
    """
    Build the ordered list of (size_annealer, label) for the node-size axis.

    First block: proportional to num_features (x1F .. x4F).
    Second block: fixed absolute sizes (10, 20, 30, 40, 50).
    Note: overlaps are intentional – they let you compare proportional vs fixed
    labelling even when the numeric value coincides.
    """
    configs = []
    for m in MULTIPLIERS:
        size = max(m * num_features, 1)   # guard against size < 1
        configs.append((size, f"x{m}F"))
    for s in FIXED_SIZES:
        configs.append((s, str(s)))
    return configs


def _prepare_data(dim: int, params: dict):
    """Generate, split, and standardise XOR data for the given dimension."""
    X, y = generate_xor_balanced(
        dim=dim,
        n_samples_dim=params['n_samples_per_region'],
        shuffle=True,
        random_seed=params['random_seed'],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['test_size'],
        random_state=params['random_seed'],
        stratify=y,
    )
    # Standardise (fit on train only – no data leakage)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std
    return X_train, X_test, y_train, y_test


def _make_loader(X_train, y_train, params: dict) -> DataLoader:
    tx = torch.tensor(X_train, dtype=torch.float32)
    ty = torch.tensor(y_train, dtype=torch.float32)
    return DataLoader(
        TensorDataset(tx, ty),
        batch_size=params['batch_size'],
        shuffle=True,
    )


def _train_one(
    num_nodi: int,
    node_size: int,
    X_train, y_train,
    X_test,  y_test,
    params: dict,
) -> dict:
    """
    Train one (num_nodi, node_size) configuration of Network_1L.

    Returns a dict with all scalar metrics and the epoch-by-epoch histories.
    """
    t0  = perf_counter()
    dev = params['device']

    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

    train_loader = _make_loader(X_train, y_train, params)
    ex = torch.tensor(X_test, dtype=torch.float32)
    ey = torch.tensor(y_test, dtype=torch.float32)

    # ── annealing settings ────────────────────────────────────────────────────
    SA_settings                  = AnnealingSettings()
    SA_settings.beta_range       = params['sa_beta_range']
    SA_settings.num_reads        = params['num_reads']
    SA_settings.num_sweeps       = params['sa_num_sweeps']
    SA_settings.num_sweeps_per_beta = params['sa_sweeps_per_beta']

    # ── model ─────────────────────────────────────────────────────────────────
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

    # ── optimizer (grouped learning rates) ────────────────────────────────────
    optim_groups = []
    for module in model.ising_perceptrons_layer:
        optim_groups += [
            {'params': [module.gamma],  'lr': params['lr_gamma']},
            {'params': [module.lmd],    'lr': params['lr_lambda']},
            {'params': [module.offset], 'lr': params['lr_offset']},
        ]
    optim_groups.append({
        'params': model.combiner_layer.parameters(),
        'lr': params['lr_classical'],
    })
    optimizer = torch.optim.Adam(optim_groups)
    loss_fn   = torch.nn.BCEWithLogitsLoss()

    # ── training loop ─────────────────────────────────────────────────────────
    training_losses   = []
    val_accuracies    = []
    best_val_acc      = 0.0
    convergence_epoch = None
    val_interval      = params['val_interval']

    for epoch in range(params['epochs']):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev).float()
            optimizer.zero_grad()
            loss = loss_fn(model(xb).view(-1), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        training_losses.append(avg_loss)

        # Validation
        do_val = (
            (epoch + 1) % val_interval == 0
            or epoch == 0
            or epoch == params['epochs'] - 1
        )
        if do_val:
            model.eval()
            with torch.no_grad():
                logits = model(ex.to(dev)).view(-1)
                probs  = torch.sigmoid(logits).cpu().numpy()
                preds  = (probs >= 0.5).astype(int)
                val_acc = accuracy_score(y_test, preds)
            val_accuracies.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if convergence_epoch is None and val_acc >= CONV_THRESHOLD:
                convergence_epoch = epoch + 1
        else:
            # repeat last known value to keep list length == epochs
            val_accuracies.append(val_accuracies[-1] if val_accuracies else 0.0)

    # ── final evaluation ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(ex.to(dev)).view(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
    preds     = (probs >= 0.5).astype(int)
    final_acc = accuracy_score(y_test, preds)
    f1        = f1_score(y_test, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = float('nan')

    final_loss = training_losses[-1] if training_losses else float('nan')
    min_loss   = min(training_losses) if training_losses else float('nan')
    # stability: std of loss in the last STABILITY_WINDOW epochs
    if len(training_losses) >= STABILITY_WINDOW:
        stability = float(np.std(training_losses[-STABILITY_WINDOW:]))
    else:
        stability = float(np.std(training_losses)) if training_losses else float('nan')

    elapsed = perf_counter() - t0

    return {
        # scalar metrics
        'accuracy':           final_acc,
        'f1':                 f1,
        'auc_roc':            auc,
        'best_val_acc':       best_val_acc,
        'final_loss':         final_loss,
        'min_loss':           min_loss,
        'loss_stability':     stability,
        'convergence_epoch':  convergence_epoch,
        'training_time_s':    elapsed,
        'n_params':           n_params,
        # epoch histories (kept for potential further analysis)
        'training_losses':    training_losses,
        'val_accuracies':     val_accuracies,
    }


# ─── plotting helpers ─────────────────────────────────────────────────────────

def _heatmap(
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    filepath: Path,
    cmap: str = 'YlOrRd',
    fmt: str = '.3f',
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = '',
):
    """Generic heatmap saver."""
    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 1.1), max(5, len(row_labels) * 0.9))
    )
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'size': 9},
        cbar_kws={'label': cbar_label} if cbar_label else {},
    )
    ax.set_xlabel('Node size  (size_annealer)', fontweight='bold')
    ax.set_ylabel('Num perceptrons  (num_nodi)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def _timing_heatmap(
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    filepath: Path,
):
    """Heatmap for training time (seconds), with string annotations."""
    annot = np.vectorize(
        lambda v: f"{v:.0f}s" if not np.isnan(v) else "ERR"
    )(matrix)
    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 1.1), max(5, len(row_labels) * 0.9))
    )
    sns.heatmap(
        matrix,
        annot=annot,
        fmt='',
        cmap='Oranges',
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'size': 9},
        cbar_kws={'label': 'seconds'},
    )
    ax.set_xlabel('Node size  (size_annealer)', fontweight='bold')
    ax.set_ylabel('Num perceptrons  (num_nodi)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ─── public API ───────────────────────────────────────────────────────────────

def run_matrix_experiment(xor_dim: int) -> dict:
    
    params    = _get_params()
    log       = _GLOBAL_LOGGER
    dim_dir   = _OUTPUT_ROOT / f"xor_{xor_dim}d"
    dim_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*80}")
    log.info(f"  MATRIX EXPERIMENT  —  XOR {xor_dim}D    [{datetime.now().strftime('%H:%M:%S')}]")
    log.info(f"{'='*80}")
    log.info(f"  Device  : {params['device']}")
    log.info(f"  Epochs  : {params['epochs']}")
    log.info(f"  Batch   : {params['batch_size']}")
    log.info(f"  LR      : γ={params['lr_gamma']}  λ={params['lr_lambda']}  "
             f"off={params['lr_offset']}  comb={params['lr_classical']}")
    log.info(f"  SA      : β={params['sa_beta_range']}  reads={params['num_reads']}  "
             f"sweeps={params['sa_num_sweeps']}")
    log.info(f"  Samples : {params['n_samples_per_region']} / region  "
             f"(test_size={params['test_size']})")

    # ── data ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = _prepare_data(xor_dim, params)
    num_features = X_train.shape[1]     # == xor_dim
    node_configs = _node_size_configs(num_features)

    log.info(f"  Train   : {len(y_train)} samples  |  Test: {len(y_test)} samples")
    log.info(f"  Grid    : {len(NUM_NODI_LIST)} × {len(node_configs)} "
             f"= {len(NUM_NODI_LIST) * len(node_configs)} configurations")

    col_labels  = [lbl for _, lbl in node_configs]
    row_labels  = [str(n) for n in NUM_NODI_LIST]
    n_rows      = len(NUM_NODI_LIST)
    n_cols      = len(node_configs)

    # Result matrices
    acc_matrix       = np.full((n_rows, n_cols), np.nan)
    f1_matrix        = np.full((n_rows, n_cols), np.nan)
    auc_matrix       = np.full((n_rows, n_cols), np.nan)
    time_matrix      = np.full((n_rows, n_cols), np.nan)
    stability_matrix = np.full((n_rows, n_cols), np.nan)
    best_val_matrix  = np.full((n_rows, n_cols), np.nan)
    conv_ep_matrix   = np.full((n_rows, n_cols), np.nan)

    detailed_records = []
    total_configs    = n_rows * n_cols
    done             = 0

    exp_start = perf_counter()

    for i, num_nodi in enumerate(NUM_NODI_LIST):
        for j, (node_size, node_label) in enumerate(node_configs):
            done += 1
            log.info(
                f"\n  [{done:3d}/{total_configs}]  "
                f"num_nodi={num_nodi:3d}  node_size={node_size:3d} ({node_label:4s})"
            )
            try:
                m = _train_one(
                    num_nodi, node_size,
                    X_train, y_train,
                    X_test,  y_test,
                    params,
                )
                acc_matrix[i, j]       = m['accuracy']
                f1_matrix[i, j]        = m['f1']
                auc_matrix[i, j]       = m['auc_roc']
                time_matrix[i, j]      = m['training_time_s']
                stability_matrix[i, j] = m['loss_stability']
                best_val_matrix[i, j]  = m['best_val_acc']
                conv_ep_matrix[i, j]   = m['convergence_epoch'] if m['convergence_epoch'] else np.nan

                log.info(
                    f"    acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                    f"auc={m['auc_roc']:.4f}  best_val={m['best_val_acc']:.4f}\n"
                    f"    min_loss={m['min_loss']:.4f}  final_loss={m['final_loss']:.4f}  "
                    f"stability={m['loss_stability']:.4f}  "
                    f"conv_epoch={m['convergence_epoch']}  "
                    f"time={m['training_time_s']:.1f}s  params={m['n_params']}"
                )

                detailed_records.append({
                    'xor_dim':           xor_dim,
                    'num_nodi':          num_nodi,
                    'node_size':         node_size,
                    'node_size_label':   node_label,
                    'accuracy':          m['accuracy'],
                    'f1':                m['f1'],
                    'auc_roc':           m['auc_roc'],
                    'best_val_acc':      m['best_val_acc'],
                    'final_loss':        m['final_loss'],
                    'min_loss':          m['min_loss'],
                    'loss_stability':    m['loss_stability'],
                    'convergence_epoch': m['convergence_epoch'],
                    'training_time_s':   m['training_time_s'],
                    'n_params':          m['n_params'],
                    'error':             '',
                })

            except Exception as e:
                log.error(f"    ERROR: {e}")
                traceback.print_exc()
                detailed_records.append({
                    'xor_dim': xor_dim, 'num_nodi': num_nodi,
                    'node_size': node_size, 'node_size_label': node_label,
                    'accuracy': np.nan, 'f1': np.nan, 'auc_roc': np.nan,
                    'best_val_acc': np.nan, 'final_loss': np.nan,
                    'min_loss': np.nan, 'loss_stability': np.nan,
                    'convergence_epoch': np.nan, 'training_time_s': np.nan,
                    'n_params': np.nan, 'error': str(e),
                })

    exp_elapsed = perf_counter() - exp_start
    log.info(f"\n  XOR {xor_dim}D grid completed in {exp_elapsed:.1f}s "
             f"({exp_elapsed/60:.1f} min)")

    # ── save CSV matrices ──────────────────────────────────────────────────────
    def _save_matrix(mat, name):
        df = pd.DataFrame(mat, index=row_labels, columns=col_labels)
        df.index.name = 'num_nodi'
        df.to_csv(dim_dir / f"{name}.csv")
        return df

    df_acc   = _save_matrix(acc_matrix,       'accuracy_matrix')
    df_f1    = _save_matrix(f1_matrix,        'f1_matrix')
    df_auc   = _save_matrix(auc_matrix,       'auc_matrix')
    df_time  = _save_matrix(time_matrix,      'timing_matrix')
    df_stab  = _save_matrix(stability_matrix, 'stability_matrix')
    df_bval  = _save_matrix(best_val_matrix,  'best_val_matrix')
    df_conv  = _save_matrix(conv_ep_matrix,   'convergence_epoch_matrix')

    # detailed per-combination log
    pd.DataFrame(detailed_records).to_csv(
        dim_dir / 'detailed_metrics.csv', index=False
    )

    # ── heatmaps ───────────────────────────────────────────────────────────────
    _heatmap(acc_matrix,  row_labels, col_labels,
             f'Accuracy — XOR {xor_dim}D',        dim_dir / 'accuracy_heatmap.png',
             cmap='YlOrRd', cbar_label='Accuracy')
    _heatmap(f1_matrix,   row_labels, col_labels,
             f'F1-Score — XOR {xor_dim}D',         dim_dir / 'f1_heatmap.png',
             cmap='Blues',  cbar_label='F1-Score')
    _heatmap(auc_matrix,  row_labels, col_labels,
             f'AUC-ROC — XOR {xor_dim}D',          dim_dir / 'auc_heatmap.png',
             cmap='Greens', cbar_label='AUC-ROC')
    _heatmap(best_val_matrix, row_labels, col_labels,
             f'Best Val Accuracy — XOR {xor_dim}D', dim_dir / 'best_val_heatmap.png',
             cmap='Purples', cbar_label='Best Val Acc')
    _timing_heatmap(time_matrix, row_labels, col_labels,
                    f'Training Time (s) — XOR {xor_dim}D', dim_dir / 'timing_heatmap.png')

    # ── console summary ────────────────────────────────────────────────────────
    log.info(f"\n{'─'*70}")
    log.info(f"  ACCURACY MATRIX  (rows=num_nodi, cols=node_size)  XOR {xor_dim}D")
    log.info(f"{'─'*70}")
    log.info(df_acc.to_string())

    if not np.all(np.isnan(acc_matrix)):
        best_idx = np.unravel_index(np.nanargmax(acc_matrix), acc_matrix.shape)
        log.info(
            f"\n  BEST CONFIG:  num_nodi={NUM_NODI_LIST[best_idx[0]]}  "
            f"node_size={col_labels[best_idx[1]]}  "
            f"→  acc={acc_matrix[best_idx]:.4f}"
        )
    log.info(f"  Output dir: {dim_dir}")

    return {
        'xor_dim':           xor_dim,
        'accuracy_matrix':   acc_matrix,
        'f1_matrix':         f1_matrix,
        'auc_matrix':        auc_matrix,
        'time_matrix':       time_matrix,
        'stability_matrix':  stability_matrix,
        'best_val_matrix':   best_val_matrix,
        'conv_epoch_matrix': conv_ep_matrix,
        'row_labels':        row_labels,
        'col_labels':        col_labels,
        'detailed_records':  detailed_records,
    }


def save_global_summary(all_results: list[dict]) -> None:
    """
    Salva il riepilogo globale su tutti i risultati raccolti finora.
    Può essere chiamata sia alla fine (con tutti e 6) sia incrementalmente.

    Output:
        summary_all_dims.csv        – una riga per ogni (dim, num_nodi, node_size)
        accuracy_grid_all_dims.png  – 2×3 pannelli, uno per dimensione XOR
    """
    log = _GLOBAL_LOGGER
    if not all_results:
        return

    # ── CSV globale ────────────────────────────────────────────────────────────
    all_records = []
    for r in all_results:
        all_records.extend(r['detailed_records'])
    df_all = pd.DataFrame(all_records)
    summary_path = _OUTPUT_ROOT / 'summary_all_dims.csv'
    df_all.to_csv(summary_path, index=False)
    log.info(f"\n  Global summary → {summary_path}")

    # ── 2×3 accuracy grid ─────────────────────────────────────────────────────
    n_panels = len(all_results)
    n_cols   = 3
    n_rows   = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 7, n_rows * 5),
        squeeze=False,
    )

    for idx, r in enumerate(all_results):
        ax   = axes[idx // n_cols][idx % n_cols]
        mat  = r['accuracy_matrix']
        mask = np.isnan(mat)
        sns.heatmap(
            mat,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=r['col_labels'],
            yticklabels=r['row_labels'],
            ax=ax,
            vmin=0, vmax=1,
            mask=mask,
            linewidths=0.3,
            annot_kws={'size': 7},
            cbar=False,
        )
        ax.set_title(f"XOR {r['xor_dim']}D", fontweight='bold', fontsize=12)
        ax.set_xlabel('Node size', fontsize=8)
        ax.set_ylabel('Num perceptrons', fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused panels
    for idx in range(len(all_results), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    fig.suptitle(
        'Accuracy Grid — Network_1L on XOR 1D–6D',
        fontweight='bold', fontsize=14, y=1.01,
    )
    plt.tight_layout()
    grid_path = _OUTPUT_ROOT / 'accuracy_grid_all_dims.png'
    plt.savefig(grid_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info(f"  Accuracy grid     → {grid_path}")

    # ── best config per dimension (console) ───────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("  GLOBAL BEST CONFIGS (by accuracy)")
    log.info(f"{'='*70}")
    log.info(f"  {'XOR dim':>7}  {'num_nodi':>9}  {'node_size':>10}  {'accuracy':>9}  {'f1':>6}  {'auc':>6}")
    log.info(f"  {'─'*7}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*6}  {'─'*6}")

    for r in all_results:
        mat = r['accuracy_matrix']
        if np.all(np.isnan(mat)):
            continue
        bi  = np.unravel_index(np.nanargmax(mat), mat.shape)
        nn  = NUM_NODI_LIST[bi[0]]
        ns  = r['col_labels'][bi[1]]
        acc = mat[bi]
        f1  = r['f1_matrix'][bi]
        auc = r['auc_matrix'][bi]
        log.info(f"  {r['xor_dim']:>7}  {nn:>9}  {ns:>10}  {acc:>9.4f}  {f1:>6.4f}  {auc:>6.4f}")

    log.info(f"\n  All outputs in: {_OUTPUT_ROOT.resolve()}")


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = []

    for dim in range(1, 7):
        result = run_matrix_experiment(dim)
        all_results.append(result)

    save_global_summary(all_results)

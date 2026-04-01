# Prompt: Genera `Inference/experiments.py`

## Contesto del progetto

Stai lavorando nel progetto **QuantumAnnealerModule**, un framework PyTorch che implementa reti neurali basate su modelli di Ising con annealing simulato. La struttura è:

```
QuantumAnnealerModule/
├── src/full_ising_model/
│   ├── full_ising_module.py    # FullIsingModule (nn.Module): parametri gamma (NxN coupling matrix), lmd (scaling), offset (bias)
│   ├── annealers.py            # AnnealingSettings, AnnealerType (SIMULATED/QUANTUM/EXACT), AnnealerFactory
│   └── utils.py                # HiddenNodesInitialization, offset(), resize_tensor(), class utils
├── ModularNetwork/
│   ├── Network_1L.py           # MultiIsingNetwork: N FullIsingModule in parallelo + combiner Linear(N,1)
│   └── Network_2L.py           # TwoLayerIsingNetwork: due strati di Ising modules
├── Inference/
│   ├── .env                    # Configurazione iperparametri
│   ├── test_matrix_xor.py      # Grid search XOR con Network_1L (num_nodi × node_size)
│   ├── test_datasetsUCI.py     # K-fold su 9 dataset UCI con FullIsingModule vs Network_1L
│   ├── test_xor.py             # XOR 1D-6D con Network_2L
│   ├── Datasets/               # 9 CSV UCI (iris, banknote, breast_cancer, ecc.)
│   └── utils/
│       ├── logger.py           # Logger(log_dir=) → info/warning/error con file + console
│       ├── plot.py             # Plot(output_dir=) → plot_loss_accuracy, plot_compare_accuracy, plot_parity_scatter, box_plot, plot_metrics_bar_comparison, plot_metrics_heatmap, plot_radar_chart, plot_combined_boxplot, plot_confusion_matrix
│       ├── utils.py            # flatten_logits, compute_metrics, save_metrics_csv, generate_xor_balanced, METRICS
│       └── dataset_manager.py  # DatasetManager: load_csv_dataset, generate_k_folds, create_dataloader
```

## API chiave da usare

### FullIsingModule (src/full_ising_model/full_ising_module.py)
```python
model = FullIsingModule(
    size_annealer=50,                    # dimensione annealer (>= input_dim, paddato automaticamente)
    annealer_type=AnnealerType.SIMULATED,
    annealing_settings=SA_settings,      # AnnealingSettings con beta_range, num_reads, num_sweeps, ecc
    lambda_init=-0.1,                    # init per parametro lmd (scaling)
    offset_init=0.0,                     # init per parametro offset (bias)
    gamma_init=None,                     # Se None: torch.randn(N,N)*0.01 triu. Altrimenti passa un tensore
    num_workers=1,
    hidden_nodes_offset_value=-0.02      # valore offset per padding degli hidden nodes nella resize
)
# Parametri learnable: model.gamma (NxN), model.lmd (scalar), model.offset (scalar)
# forward(thetas) → scalar energy per sample
```

### MultiIsingNetwork (ModularNetwork/Network_1L.py)
```python
model = MultiIsingNetwork(
    num_ising_perceptrons=5,             # numero di FullIsingModule in parallelo
    size_annealer=50,
    annealing_settings=SA_settings,
    annealer_type=AnnealerType.SIMULATED,
    lambda_init=-0.1,
    offset_init=0.0,
    partition_input=False                # se True, partiziona l'input tra i perceptron
)
# model.ising_perceptrons_layer → ModuleList di FullIsingModule
# model.combiner_layer → Linear(num_perceptrons, 1)
# forward(thetas) → scalar per sample
```

### AnnealingSettings
```python
SA_settings = AnnealingSettings()
SA_settings.beta_range = [1, 10]
SA_settings.num_reads = 1
SA_settings.num_sweeps = 1000
SA_settings.num_sweeps_per_beta = 1
```

### Utilities già disponibili
```python
from Inference.utils.logger import Logger
from Inference.utils.plot import Plot
from Inference.utils.utils import flatten_logits, compute_metrics, save_metrics_csv, generate_xor_balanced, METRICS
from Inference.utils.dataset_manager import DatasetManager
```

- `flatten_logits(logits)` → 1D tensor (gestisce output shape (N,) o (N,1))
- `compute_metrics(y_true, probs)` → dict con accuracy, precision, recall, f1, auc
- `generate_xor_balanced(dim, n_samples_dim=100, shuffle=True, random_seed=42)` → (X, y) numpy
- `DatasetManager().load_csv_dataset(path)` → (X, y) numpy
- `Plot(output_dir=...)` ha metodi: `plot_loss_accuracy(losses, accs, name)`, `plot_compare_accuracy(m1, m2, labels)`, `plot_parity_scatter(...)`, `box_plot(...)`, `plot_metrics_bar_comparison(...)`, `plot_metrics_heatmap(...)`, `plot_radar_chart(...)`, `plot_combined_boxplot(...)`, `plot_confusion_matrix(y_true, y_pred, name)`
- `save_metrics_csv(names, metrics1, metrics2, out_dir, model_name1=..., model_name2=...)` → csv_path

### Come funziona la resize/padding con hidden_nodes_offset_value
Quando `size_annealer > input_dim`, il FullIsingModule fa padding automatico con la funzione `offset`:
- Ripete l'input ciclicamente e aggiunge `hidden_nodes_offset_value * (indice_ripetizione)` 
- Es: theta=[0.5, 0.3], size=4, offset=-0.02 → [0.5, 0.3, 0.48, 0.28]
- Il parametro `hidden_nodes_offset_value` nel costruttore di FullIsingModule controlla questo

### Come funziona gamma
- `gamma` è una matrice NxN upper-triangular (coupling J_ij nell'Ising)
- Default init: `torch.randn(N,N) * 0.01` poi `triu(diagonal=1)`
- Si può passare un tensore custom via `gamma_init` al costruttore
- I gradienti sono calcolati via outer product delle configurazioni di spin campionate

### Pattern di training standard (da copiare)
```python
# Optimizer con learning rate separate per ogni parametro
optimizer = torch.optim.Adam([
    {'params': [model.gamma], 'lr': 0.001},
    {'params': [model.lmd], 'lr': 0.005},
    {'params': [model.offset], 'lr': 0.01},
])
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = flatten_logits(model(xb))
        loss = loss_fn(pred, yb.float())
        loss.backward()
        optimizer.step()
    # Validazione
    model.eval()
    with torch.no_grad():
        logits = flatten_logits(model(X_test_tensor))
        probs = torch.sigmoid(logits).cpu().numpy()
        acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
```

Per MultiIsingNetwork, l'optimizer ha gruppi separati per ogni perceptron + combiner:
```python
optim_groups = []
for module in model.ising_perceptrons_layer:
    optim_groups += [
        {'params': [module.gamma],  'lr': lr_gamma},
        {'params': [module.lmd],    'lr': lr_lambda},
        {'params': [module.offset], 'lr': lr_offset},
    ]
optim_groups.append({'params': model.combiner_layer.parameters(), 'lr': lr_combiner})
optimizer = torch.optim.Adam(optim_groups)
```

---

## Cosa devi generare

Crea il file `Inference/experiments.py` contenente **4 esperimenti** indipendenti, ciascuno come funzione separata. Il `main` lancia tutti e 4 gli esperimenti **in parallelo** usando `multiprocessing.Process`. Ogni esperimento salva i propri risultati (grafici PNG, log, CSV) in una sottocartella dedicata dentro `experiments/experiments_YYYYMMDD_HHMMSS/`.

### Struttura output:
```
experiments/experiments_YYYYMMDD_HHMMSS/
├── exp1_matrix_fullising_vs_net1l/
│   ├── run_*.log
│   ├── xor_1d/, xor_2d/, xor_3d/     (heatmap accuracy, f1, auc, tempo)
│   ├── iris/                           (heatmap accuracy, f1, auc, tempo)
│   ├── summary.csv
│   └── *.png
├── exp2_offset_sweep/
│   ├── run_*.log
│   ├── xor_1d/, xor_2d/, xor_3d/
│   ├── per ogni dataset UCI/
│   ├── summary.csv
│   └── *.png
├── exp3_gamma_init/
│   ├── run_*.log
│   ├── xor_1d/, xor_2d/, xor_3d/
│   ├── summary.csv
│   └── *.png
└── exp4_staged_training/
    ├── run_*.log
    ├── xor_1d/, xor_2d/, xor_3d/
    ├── summary.csv
    └── *.png
```

---

## Esperimento 1: Matrix FullIsing vs Network1L (con SGD e MSELoss)

**Nome:** `exp1_matrix_fullising_vs_net1l`
**Descrizione:** "Grid search su XOR 1D-3D e Iris: confronto FullIsingModule vs MultiIsingNetwork con SGD e MSELoss"

Per ogni dimensione XOR (1, 2, 3) e per il dataset Iris, esegui una griglia di configurazioni:
- **Righe:** `num_nodi` = [1, 2, 3, 5, 8] (per Network1L; per FullIsingModule usa un singolo modulo)
- **Colonne:** `size_annealer` = [x1F, x2F, x3F, x4F, 10, 20, 30, 50] dove xNF = N * num_features

Per **ogni cella** della griglia allena sia FullIsingModule che MultiIsingNetwork.

**Iperparametri:**
- Optimizer: `torch.optim.SGD` (NON Adam come negli altri test)
- Loss: `torch.nn.MSELoss()` (NON BCEWithLogitsLoss). Poiché MSELoss lavora su valori continui, applica `torch.sigmoid(pred)` prima di calcolare la loss: `loss = mse_loss(torch.sigmoid(pred), target)`
- Learning rate SGD: `lr_gamma=0.01, lr_lambda=0.01, lr_offset=0.05, lr_combiner=0.01`
- Momentum SGD: 0.9
- Epochs: 150
- Batch size: 32
- `lambda_init=-0.1, offset_init=0.0`
- SA: beta_range=[1,10], num_reads=1, num_sweeps=1000, sweeps_per_beta=1
- `n_samples_per_region=100`, test_size=0.2
- random_seed=42

**Output per ogni dimensione/dataset:**
- Heatmap accuracy (2 heatmap affiancate: FullIsing e Net1L)
- Heatmap F1, AUC, tempo di training
- CSV con tutte le metriche dettagliate
- Log completo

**Per FullIsingModule** nella griglia: ignora la riga `num_nodi`, usa solo la colonna `size_annealer`. Genera una riga singola "FullIsing" nella heatmap.

### Note implementative:
- Per Iris: carica con `DatasetManager().load_csv_dataset('Inference/Datasets/00_iris_versicolor_virginica.csv')`
- Non fare K-fold per questo esperimento, usa un singolo split train/test con stratify
- Usa le funzioni heatmap dal pattern di `test_matrix_xor.py` (riscrivile in experiments.py, copia il codice)
- Per i grafici usa seaborn/matplotlib come in test_matrix_xor.py

---

## Esperimento 2: Offset Sweep (hidden_nodes_offset_value = 1/node_size)

**Nome:** `exp2_offset_sweep`
**Descrizione:** "Effetto dell'offset = 1/size_annealer per padding hidden nodes su XOR 1D-3D e tutti i dataset UCI"

Confronta FullIsingModule e MultiIsingNetwork con `hidden_nodes_offset_value = 1/size_annealer` rispetto al default (`-0.02`).

**Procedura:**
Per ogni dataset (XOR 1D, 2D, 3D + tutti i 9 CSV UCI):
1. Allena FullIsingModule con `hidden_nodes_offset_value=-0.02` (default)
2. Allena FullIsingModule con `hidden_nodes_offset_value=1/size_annealer`
3. Allena MultiIsingNetwork con `hidden_nodes_offset_value=-0.02` (default)
4. Allena MultiIsingNetwork con `hidden_nodes_offset_value=1/size_annealer`

**Come settare hidden_nodes_offset_value personalizzato:**
Il parametro `hidden_nodes_offset_value` è passato direttamente al costruttore di FullIsingModule. Per MultiIsingNetwork, devi settarlo su ogni modulo dopo la creazione:
```python
model = MultiIsingNetwork(...)
for module in model.ising_perceptrons_layer:
    module.hidden_nodes_config.fun_args = (1.0 / size_annealer,)
    module.hidden_nodes_config._resize_cache = {}  # invalida la cache
```

**Iperparametri:**
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- lr_gamma=0.001, lr_lambda=0.005, lr_offset=0.01, lr_combiner=0.005
- Epochs: 200
- Batch size: 32
- size_annealer: 50 (fisso)
- num_ising_perceptrons: 5 (per Net1L)
- lambda_init=-0.1, offset_init=0.0
- SA: beta_range=[1,10], num_reads=1, num_sweeps=1000, sweeps_per_beta=1
- K-Folds: 5 (per i dataset UCI), singolo split per XOR
- random_seed=42

**Output:**
- Per ogni dataset: grafico confronto 4 modelli (bar chart con errbar se K-fold)
- Plot loss/accuracy per ogni configurazione
- Tabella CSV riassuntiva con metriche per tutti i dataset × tutte le configurazioni
- Heatmap metriche: righe=dataset, colonne=4 varianti (FullIsing_default, FullIsing_1/N, Net1L_default, Net1L_1/N)
- Radar chart globale (media su tutti i dataset) confrontando le 4 varianti
- Box plot K-fold per i dataset UCI

---

## Esperimento 3: Gamma Initialization Strategies

**Nome:** `exp3_gamma_init`
**Descrizione:** "Confronto strategie di inizializzazione di gamma su XOR 1D-3D"

Testa 5 strategie di inizializzazione di gamma su XOR 1D-3D, sia per FullIsingModule che per MultiIsingNetwork:

1. **`zeros`**: `gamma_init = torch.zeros(N, N)`
2. **`small_randn`**: `gamma_init = torch.triu(torch.randn(N,N) * 0.001, diagonal=1)`
3. **`medium_randn`**: `gamma_init = torch.triu(torch.randn(N,N) * 0.1, diagonal=1)` 
4. **`large_randn`**: `gamma_init = torch.triu(torch.randn(N,N) * 1.0, diagonal=1)`
5. **`theta_ratio`**: Ogni nodo del grafo Ising contiene il valore di una feature (theta). L'arco gamma[i,j] che collega due nodi deve essere il rapporto tra i loro valori. In pratica:
   - Calcola il vettore theta rappresentativo: media delle feature del training set (standardizzato), di dimensione `input_dim`
   - Estendilo a `size_annealer` con la stessa logica di padding usata dal modello (repeat ciclico con offset)
   - Per ogni coppia (i,j) con i < j: `gamma[i,j] = theta_padded[j] / (theta_padded[i] + eps)` dove `eps=1e-8` per evitare divisione per zero
   - Applica `triu(diagonal=1)` per tenere solo la parte upper-triangular
   
   Codice di riferimento per costruire theta_padded:
   ```python
   # theta medio dal training set (dopo standardizzazione)
   mean_theta = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)  # shape: (input_dim,)
   
   # padding a size_annealer: stessa logica della funzione offset() in utils.py
   # theta_padded[k] = mean_theta[k % input_dim] + (k // input_dim) * hidden_nodes_offset_value
   indices = torch.arange(size_annealer)
   theta_padded = mean_theta[indices % len(mean_theta)] + (indices // len(mean_theta)).float() * (-0.02)
   
   # gamma[i,j] = theta_padded[j] / theta_padded[i]  (rapporto colonna / riga)
   eps = 1e-8
   gamma_init = theta_padded.unsqueeze(0) / (theta_padded.unsqueeze(1) + eps)  # shape (N,N)
   gamma_init = torch.triu(gamma_init, diagonal=1)
   ```

**Come passare gamma_init:**
```python
# Per FullIsingModule:
model = FullIsingModule(size_annealer=N, ..., gamma_init=custom_gamma)

# Per MultiIsingNetwork: crea il modello normalmente, poi sovrascrivi i gamma
model = MultiIsingNetwork(...)
for module in model.ising_perceptrons_layer:
    with torch.no_grad():
        module.gamma.copy_(custom_gamma)
```

**Iperparametri:**
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- lr_gamma=0.001, lr_lambda=0.005, lr_offset=0.01, lr_combiner=0.005
- Epochs: 200
- Batch size: 32
- size_annealer: 50
- num_ising_perceptrons: 5
- lambda_init=-0.1, offset_init=0.0
- SA: beta_range=[1,10], num_reads=1, num_sweeps=1000, sweeps_per_beta=1
- random_seed=42

**Output per dimensione XOR:**
- Grafico curva loss training per tutte le 5 inizializzazioni sovrapposte (sia FullIsing che Net1L in subplot separati)
- Grafico curva accuracy validazione per tutte le 5 inizializzazioni sovrapposte
- Bar chart accuracy finale per le 5 inizializzazioni × 2 modelli
- Heatmap: righe=inizializzazione, colonne=metriche (acc, f1, auc), per ogni modello
- CSV con tutte le metriche
- Grafico convergenza: epoca a cui si raggiunge il 90% dell'accuracy finale

---

## Esperimento 4: Staged Training (Freeze & Unfreeze)

**Nome:** `exp4_staged_training`
**Descrizione:** "Training a 2 fasi: prima solo gamma, poi gamma (lr ridotto) + lambda + offset su XOR 1D-3D"

Per XOR 1D-3D, sia con FullIsingModule che MultiIsingNetwork:

**Fase 1 (epoche 1 → N_phase1):**
- Allena **solo gamma**, congela lambda e offset
- `model.lmd.requires_grad = False`
- `model.offset.requires_grad = False`
- (Per MultiIsingNetwork: fai lo stesso per ogni modulo + congela combiner_layer)
- lr_gamma = 0.005

**Fase 2 (epoche N_phase1+1 → N_total):**
- Scongela lambda e offset: `model.lmd.requires_grad = True`, `model.offset.requires_grad = True`
- Abbassa lr_gamma: `lr_gamma = 0.0005` (10x meno)
- lr_lambda = 0.005, lr_offset = 0.01
- (Per MultiIsingNetwork: scongela anche combiner_layer con lr=0.005)
- **Ricrea l'optimizer** con i nuovi gruppi di parametri (non basta cambiare lr a runtime)

**Come freezare/scongelere per MultiIsingNetwork:**
```python
# Freeze fase 1:
for module in model.ising_perceptrons_layer:
    module.lmd.requires_grad = False
    module.offset.requires_grad = False
for p in model.combiner_layer.parameters():
    p.requires_grad = False

# Optimizer fase 1 (solo gamma):
optim_groups = [{'params': [m.gamma], 'lr': 0.005} for m in model.ising_perceptrons_layer]
optimizer = torch.optim.Adam(optim_groups)

# Unfreeze fase 2:
for module in model.ising_perceptrons_layer:
    module.lmd.requires_grad = True
    module.offset.requires_grad = True
for p in model.combiner_layer.parameters():
    p.requires_grad = True

# Optimizer fase 2:
optim_groups = []
for module in model.ising_perceptrons_layer:
    optim_groups += [
        {'params': [module.gamma],  'lr': 0.0005},
        {'params': [module.lmd],    'lr': 0.005},
        {'params': [module.offset], 'lr': 0.01},
    ]
optim_groups.append({'params': model.combiner_layer.parameters(), 'lr': 0.005})
optimizer = torch.optim.Adam(optim_groups)
```

**Baseline di confronto:** per ogni dimensione, allena anche senza staging (training standard per tutte le epoche) come baseline.

**Iperparametri:**
- N_phase1: 100 epoche
- N_total: 250 epoche (fase 2 = epoche 101-250)
- Loss: BCEWithLogitsLoss
- Batch size: 32
- size_annealer: 50
- num_ising_perceptrons: 5
- lambda_init=-0.1, offset_init=0.0
- SA: beta_range=[1,10], num_reads=1, num_sweeps=1000, sweeps_per_beta=1
- random_seed=42

**Output per dimensione:**
- Grafico loss con linea verticale a epoch=N_phase1 che segna il cambio di fase (3 curve: staged_FullIsing, staged_Net1L, baseline)
- Grafico accuracy con linea verticale
- Bar chart accuracy finale: staged vs baseline × 2 modelli (4 barre)
- CSV con metriche finali e metriche a fine fase 1
- Log con dettagli per-epoca

---

## Requisiti generali per il file experiments.py

### Imports
```python
import os, sys, traceback, json
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing
import dotenv

dotenv.load_dotenv(dotenv_path=Path(__file__).parent / '.env')

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.utils.logger import Logger
from Inference.utils.plot import Plot
from Inference.utils.utils import flatten_logits, compute_metrics, save_metrics_csv, generate_xor_balanced, METRICS
from Inference.utils.dataset_manager import DatasetManager
from full_ising_model.full_ising_module import FullIsingModule
from full_ising_model.annealers import AnnealingSettings, AnnealerType
from ModularNetwork.Network_1L import MultiIsingNetwork
```

### Struttura del main
```python
if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"experiments/experiments_{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("exp1_matrix_fullising_vs_net1l", run_exp1),
        ("exp2_offset_sweep", run_exp2),
        ("exp3_gamma_init", run_exp3),
        ("exp4_staged_training", run_exp4),
    ]

    processes = []
    for name, func in experiments:
        exp_dir = base_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        p = multiprocessing.Process(target=func, args=(exp_dir,), name=name)
        processes.append(p)
        p.start()
        print(f"[MAIN] Avviato esperimento: {name} (PID {p.pid})")

    for p in processes:
        p.join()
        status = "OK" if p.exitcode == 0 else f"ERRORE (exit={p.exitcode})"
        print(f"[MAIN] Completato: {p.name} → {status}")

    print(f"\n[MAIN] Tutti gli esperimenti completati. Risultati in: {base_dir.resolve()}")
```

### Funzioni helper comuni (definiscile in cima al file)
- `make_sa_settings(beta_range, num_reads, num_sweeps, sweeps_per_beta)` → AnnealingSettings
- `prepare_xor_data(dim, n_samples, test_size, seed)` → X_train, X_test, y_train, y_test (standardizzati)
- `make_loader(X_train, y_train, batch_size)` → DataLoader
- `heatmap(matrix, row_labels, col_labels, title, filepath, ...)` → copia dalla funzione `_heatmap` di test_matrix_xor.py
- `timing_heatmap(...)` → copia da test_matrix_xor.py

### Convenzioni
- Ogni funzione `run_expN(exp_dir: Path)` è self-contained
- Ogni esperimento crea il proprio Logger con `log_dir=str(exp_dir)`
- Usa `torch.manual_seed(seed)` e `np.random.seed(seed)` prima di ogni training
- Standardizza dati: fit su train only, trasforma train e test
- Gestisci std=0 con `std[std == 0] = 1e-8`
- Salva tutti i grafici a 300 DPI con `bbox_inches='tight', facecolor='white'`
- Usa try/except per ogni singola configurazione: se un config fallisce, logga l'errore e continua
- Ogni training logga periodicamente (ogni 20 epoche): epoch, loss, accuracy

### NON fare
- Non usare cuda (il codice fa annealing su CPU)
- Non importare da test_matrix_xor.py o test_datasetsUCI.py — copia le funzioni helper necessarie direttamente in experiments.py
- Non creare file aggiuntivi fuori da experiments.py (tutto in un file)
- Non modificare file esistenti del progetto

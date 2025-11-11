import os
import glob
import time
import argparse
import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Ensure repository root is on sys.path so sibling packages import correctly when
# running this script from the Inference/ folder.
import sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from utils import (
    load_csv_dataset,
    prepare_datasets,
    plot_training_loss,
    plot_confusion_matrix,
    plot_function_predictions,
    generate_xor_balanced,
    fquad,
    flog,
    get_next_plot_filename,
    plot_results_table_scientific,
)

from NeuralNetworkIsing.data_ import SimpleDataset, HiddenNodesInitialization
from NeuralNetworkIsing.NeuralNetIsing import MultiIsingNetwork
from TorchIsingModule.IsingModule import FullIsingModule
from IsingModule.ising_learning_model.sim_anneal_model import AnnealingSettings


def _env(name, default=None, parse=True):
    val = os.getenv(name)
    if val is None:
        return default
    if parse:
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val


def train_single_model(X_train, y_train, X_val, y_val, config):
    # prepare datasets and loaders using shared util
    dataset, test_set, train_loader, test_loader = prepare_datasets(
        X_train, y_train, X_val, y_val, input_dim=config['input_dim']
    )

    SA_settings = AnnealingSettings()
    SA_settings.beta_range = config.get('SA_beta_range', [1, 10])
    SA_settings.num_reads = config.get('SA_num_reads', 1)
    SA_settings.num_sweeps = config.get('SA_num_sweeps', 100)
    SA_settings.sweeps_per_beta = config.get('SA_sweeps_per_beta', 1)

    model = FullIsingModule(config['size'], SA_settings, config['lambda_init'], config['offset_init']).to(config['device'])

    optimizer = torch.optim.SGD([
        {'params': [model.ising_layer.gamma], 'lr': config['lr_gamma']},
        {'params': [model.lmd], 'lr': config['lr_lambda']},
        {'params': [model.offset], 'lr': config['lr_offset']},
    ])

    if config['task'] == 'class':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    training_losses = []
    for epoch in range(config['epochs']):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(config['device'])
            y_batch = y_batch.to(config['device'])
            optimizer.zero_grad()
            pred = model(x_batch).view(-1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
        training_losses.append(loss.item())

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_set.x.to(config['device'])).cpu().view(-1)
        preds = preds_tensor.numpy()

    if config['task'] == 'class':
        classes = config['classes']
        predictions = np.where(preds < 0.5, classes[0], classes[1])
        return training_losses, predictions, preds
    else:
        return training_losses, preds, preds


def train_neural_net(X_train, y_train, X_val, y_val, config):
    # Use SimpleDataset resizing and hidden nodes init similarly to other scripts
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

    hn = HiddenNodesInitialization(config.get('HN_init', 'function'))
    # Optionally set function on hn if provided
    if hasattr(config, 'get') and config.get('HN_function'):
        hn.function = config.get('HN_function')
    hn.fun_args = config.get('HN_fun_args', None)

    if config.get('partition_input'):
        dataset.resize(config['size'] * config['num_ising_perceptrons'], hn)
        test_set.resize(config['size'] * config['num_ising_perceptrons'], hn)
    else:
        dataset.resize(config['size'], hn)
        test_set.resize(config['size'], hn)

    train_loader = DataLoader(TensorDataset(dataset.x, dataset.y), batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_set.x, test_set.y), batch_size=config['batch_size'], shuffle=False)

    model = MultiIsingNetwork(
        num_ising_perceptrons=config['num_ising_perceptrons'],
        sizeAnnealModel=config['size'],
        anneal_settings=AnnealingSettings(),
        lambda_init=config['lambda_init'],
        offset_init=config['offset_init'],
        partition_input=config.get('partition_input', False),
    ).to(config['device'])

    optimizer_grouped_parameters = []
    for p_idx, single_module in enumerate(model.ising_perceptrons_layer):
        optimizer_grouped_parameters.append({'params': [single_module.ising_layer.gamma], 'lr': config['lr_gamma']})
        optimizer_grouped_parameters.append({'params': [single_module.lmd], 'lr': config['lr_lambda']})
        optimizer_grouped_parameters.append({'params': [single_module.offset], 'lr': config['lr_offset']})
    optimizer_grouped_parameters.append({'params': model.combiner_layer.parameters(), 'lr': config.get('lr_combiner', 0.01)})

    optimizer = torch.optim.SGD(optimizer_grouped_parameters)
    loss_fn = torch.nn.MSELoss() if config['task'] == 'regr' else torch.nn.BCEWithLogitsLoss()

    # Training
    training_losses = model.train_model(train_loader, optimizer, loss_fn, config['epochs'], config['device'], print_every=10)

    # Testing
    predictions_test, targets_test = model.test(test_loader, config['device'])
    if config['task'] == 'class':
        classes = config['classes']
        predictions = np.where(predictions_test < (abs(classes[1]) - abs(classes[0])) / 2, classes[0], classes[1])
        return training_losses, predictions, predictions_test, targets_test
    else:
        return training_losses, predictions_test, predictions_test, targets_test


def run_pipeline(args):
    # load configuration from .env
    classes = _env('CLASSES', [0, 1])
    batch_size = int(_env('BATCH_SIZE', 32))
    epochs = int(_env('EPOCHS', 50))
    size = int(_env('SIZE', 10))
    num_isings = int(_env('NUM_ISING_PERCEPTRONS', 1))
    device = torch.device('cuda' if torch.cuda.is_available() and _env('DEVICE', 'cpu') == 'cuda' else 'cpu')

    config = {
        'classes': classes,
        'batch_size': batch_size,
        'epochs': epochs,
        'size': size,
        'num_ising_perceptrons': num_isings,
        'lambda_init': float(_env('LAMBDA_INIT', -0.01)),
        'offset_init': float(_env('OFFSET_INIT', 0)),
        'lr_gamma': float(_env('LEARNING_RATE_GAMMA', 0.05)),
        'lr_lambda': float(_env('LEARNING_RATE_LAMBDA', 0.01)),
        'lr_offset': float(_env('LEARNING_RATE_OFFSET', 0.01)),
        'lr_combiner': float(_env('LR_COMBINER', 0.01)),
        'device': device,
        'task': args.task,
        'input_dim': None,
        'partition_input': _env('PARTITION_INPUT', False),
        'HN_init': _env('HN_init', 'function'),
        'HN_fun_args': _env('HN_fun_args', None),
    }

    if args.dataset == 'xor':
        # generate XOR dataset
        if args.task == 'class':
            X, y = generate_xor_balanced(int(_env('DATA_INPUT_DIM', 2)), n_samples_dim=int(_env('TRAINING_SAMPLES', 200)))
            # split
            split = int(len(y) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            # regression: use fquad or flog based on env FUNC
            func_name = _env('FUNC', None)
            func = {'fquad': fquad, 'flog': flog}.get(func_name, fquad)
            X = np.linspace(_env('RANGES_TRAIN', [[0, 1]])[0][0], _env('RANGES_TRAIN', [[0, 1]])[0][1], int(_env('NUM_SAMPLES_TRAIN', 200)))
            y = func(X)
            X = X.reshape(-1, 1)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

    else:
        # assume dataset path or directory
        if args.dataset and os.path.isfile(args.dataset):
            X, y = load_csv_dataset(args.dataset)
            # simple train/val 80/20 split
            split = int(len(y) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            # iterate over csvs in datasets/ like test suits
            csvs = glob.glob(os.path.join(os.path.dirname(__file__), 'datasets', '*.csv'))
            if not csvs:
                raise RuntimeError('No dataset CSVs found and no --dataset provided')
            X, y = load_csv_dataset(csvs[0])
            split = int(len(y) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

    config['input_dim'] = X_train.shape[1] if X_train.ndim > 1 else 1

    start = time.time()
    if args.model == 'single':
        training_losses, preds, raw = train_single_model(X_train, y_train, X_val, y_val, config)
        if args.task == 'class':
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average='binary')
            print(f'SingleModel classification acc={acc:.4f} f1={f1:.4f}')
            plot_training_loss(training_losses, 'SingleModel')
            plot_confusion_matrix(y_val, preds, labels=config['classes'], save_path=get_next_plot_filename('confusion_single', 'result', 'png'))
        else:
            # regression: plot function vs preds
            try:
                func = fquad
                plot_function_predictions(X_val.flatten(), raw, func=func, ranges=_env('RANGES_TEST', [[0, 1]]), save_path=get_next_plot_filename('regr_single', 'result', 'png'))
            except Exception:
                pass
    else:
        training_losses, preds, raw_preds, targets = train_neural_net(X_train, y_train, X_val, y_val, config)
        if args.task == 'class':
            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average='binary')
            print(f'NeuralNet classification acc={acc:.4f} f1={f1:.4f}')
            plot_training_loss(training_losses, 'NeuralNet')
            plot_confusion_matrix(targets, preds, labels=config['classes'], save_path=get_next_plot_filename('confusion_neural', 'result', 'png'))
        else:
            # regression plot
            plot_function_predictions(X_val.flatten(), raw_preds, func=fquad, ranges=_env('RANGES_TEST', [[0, 1]]), save_path=get_next_plot_filename('regr_neural', 'result', 'png'))

    end = time.time()
    print(f'Elapsed: {end - start:.2f}s')


def build_argparser():
    p = argparse.ArgumentParser(description='Unified test runner for Ising models')
    p.add_argument('--model', choices=['single', 'neural'], default='single', help='Which model to run')
    p.add_argument('--task', choices=['class', 'regr'], default='class', help='Classification or regression')
    p.add_argument('--dataset', default='xor', help="Path to CSV dataset or 'xor' to generate XOR/synthetic data")
    return p


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)

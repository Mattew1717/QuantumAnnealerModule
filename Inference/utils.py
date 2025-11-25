import numpy as np

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
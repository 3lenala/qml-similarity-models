from itertools import combinations
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

SEED = 42
SAMPLES = 250

class DataBase:
    def __init__(self, X, y):
        self.features = jnp.array(X, dtype=jnp.float64)
        self.labels = jnp.array(y, dtype=jnp.float64)

    def split_raw_data(self):
        xs_train, xs_test, ys_train, ys_test = train_test_split(
            self.features, self.labels, test_size=0.5, random_state=SEED)
        xs_val, xs_test, ys_val, ys_test = train_test_split(
            xs_test, ys_test, test_size=0.5, random_state=SEED)
        return {'train': DataBase(xs_train, ys_train), 'val': DataBase(xs_val, ys_val), 'test': DataBase(xs_test, ys_test)}

    def build_pairs(self):
        def create_pairs(X, y):
            n = len(X)
            pairs, labels = [], []
            for i, j in combinations(range(n), 2):
                pairs.append([X[i].tolist(), X[j].tolist()])
                labels.append(1 if y[i] == y[j] else 0)
            return pairs, labels
        X_pairs, y_pairs = create_pairs(self.features, self.labels)
        return DataBase(X_pairs, y_pairs)
    

class SplittedDb:
    def __init__(self, database):
        sets = database.split_raw_data()
        self.train = sets['train']
        self.val = sets['val']
        self.test = sets['test']


class PairsSet:
    def __init__(self, X, y):
        self.original = DataBase(X, y)
        self.split = SplittedDb(self.original)
        self.train = self.split.train.build_pairs()
        self.len_train = len(self.train.labels)
        self.val = self.split.val.build_pairs()
        self.test = self.split.test.build_pairs()



def sinus_dataset_generator(samples: int, a: float = 0.8, seed: int = SEED):
    rng = np.random.default_rng(seed)

    n_per_class = samples // 2
    X, y = [], []
    count = [0, 0]

    while count[0] < n_per_class or count[1] < n_per_class:
        x = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        label = int(x[1] < -a * np.sin(np.pi * x[0]))
        if count[label] < n_per_class:
            X.append(x)
            y.append(label)
            count[label] += 1

    X = np.array(X)
    # y = 2 * np.array(y) - 1
    y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def plot_set(dataset: SplittedDb, sinus:bool=True):

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        ('train', 0): '#39568CFF', ('train', 1): '#73D055FF',
        ('val', 0):   '#2D708EFF', ('val', 1):   '#B8De29ff',
        ('test', 0):  '#238A8DFF', ('test', 1):  '#FDE725FF',
    }

    splits = [
        ('train', dataset.train.features, dataset.train.labels, 'o', 0.9),
        ('val',   dataset.val.features, dataset.val.labels,   '^', 0.9),
        ('test',  dataset.test.features, dataset.test.labels,  's', 0.9),
    ]

    for split_name, X, y, marker, alpha in splits:
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(
                X[mask, 0], X[mask, 1],
                c=colors[(split_name, cls)],
                marker=marker, alpha=alpha,
                s=70 if split_name != 'train' else 50,
                edgecolors='black',
                linewidths=0.6,
                label=f'{split_name} - class {cls}'
            )

    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    if sinus:
        x_range = np.linspace(-1.0, 1.0, 300)
        ax.plot(x_range, -0.8 * np.sin(np.pi * x_range), color='black', linestyle='--',
        linewidth=1.5, label=f'Decision frontier: y=-{0.8}·sin(πx)', zorder=5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig('split_visualization.png', dpi=150)
    plt.close()


X, y = sinus_dataset_generator(samples=SAMPLES, seed=SEED)
database = DataBase(X,y)
sinus = SplittedDb(database)
sinus_pairs = PairsSet(X, y)
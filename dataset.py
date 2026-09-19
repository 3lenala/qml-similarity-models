from itertools import combinations
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


jax.config.update('jax_enable_x64', True)
#enables 64 bit float numbers

SEED = 42
SAMPLES = 500

class DataBase:
    """Stores a database consisting of single input feature vectors and their respective labels.
    Includes operations that you can apply to it: 
                - Separate data into train/val/test
                - Create all possible pairs"""
    def __init__(self, X, y, pair_types=None):
        self.features = jnp.array(X, dtype=jnp.float64)
        self.labels = jnp.array(y, dtype=jnp.float64)
        self.pair_types = pair_types

    def split_raw_data(self):
        xs_train, xs_test, ys_train, ys_test = train_test_split(
            self.features, self.labels, test_size=0.5, random_state=SEED)
        xs_val, xs_test, ys_val, ys_test = train_test_split(
            xs_test, ys_test, test_size=0.5, random_state=SEED)
        return {'train': DataBase(xs_train, ys_train), 'val': DataBase(xs_val, ys_val), 'test': DataBase(xs_test, ys_test)}

def build_pairs(self):
    def create_pairs(X, y):
        n = len(X)
        pairs = []
        labels = []
        pair_types = []

        for i, j in combinations(range(n), 2):

            pairs.append([
                X[i].tolist(),
                X[j].tolist()
            ])

            labels.append(
                1 if y[i] == y[j] else 0
            )

            # (0,1) y (1,0) -> are both the same case "0-1"
            c1 = int(y[i])
            c2 = int(y[j]) 
            pair_type = tuple(sorted((c1, c2)))

            pair_types.append(pair_type)

        return pairs, labels, pair_types

    X, y, pair_types = create_pairs(
        self.features,
        self.labels
    )

    return DataBase(X, y, pair_types)

class SplittedDb:
    """Stores a dataset splitted into train/val/test"""
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

        val_feat, threshold_val_feat, val_labels, threshold_val_labels = train_test_split(
            self.split.val.features,
            self.split.val.labels,
            test_size=0.3,
            random_state=SEED
        )

        self.val = DataBase(
            val_feat,
            val_labels
        ).build_pairs()

        self.threshold_val = DataBase(
            threshold_val_feat,
            threshold_val_labels
        ).build_pairs()

        self.test = self.split.test.build_pairs()
        
class MultiClassSet:
    def __init__(self,X,y):
        self.original = DataBase(X,y)
        self.X, self.y, self.X2, self.y2 = self.separate()
        pairs = PairsSet(self.X, self.y)
        self.train = pairs.train
        self.len_train = len(self.train.features)
        self.val = pairs.val
        self.test = self.create_test_pairs(pairs.split.test.features)
    
    def separate(self):
        mask = self.original.labels == 2
        return self.original.features[~mask], self.original.labels[~mask], self.original.features[mask], self.original.labels[mask]
    
    def create_test_pairs(self, X):
        pairs, labels = [], []
        n, m = max(len(X), len(self.X2)), min(len(X), len(self.X2)) 
        for i in range(len(X)):
            for j in range(len(self.X2)):
                pairs.append([X[i].tolist(), self.X2[j].tolist()])
                labels.append(0)
        for i, j in combinations(range(len(self.X2)),2):
            pairs.append([self.X2[i].tolist(), self.X2[j].tolist()])
            labels.append(1)          
        return DataBase(pairs, labels)


def sinus_dataset_generator(samples: int, a: float = 0.8, seed: int = SEED):
    """Code taken from Pablo Rodriguez-Grasa, Yue Ban, and Mikel Sanz
      Neural quantum kernels: Training quantum kernels with quantum 
      neural networks(2025)
       Generates a dataset of two possible classes divided by the decision 
        frontier: x1=-a·sin(π x0) for 'a' a configurable parameter"""
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
    y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def sinus_dataset_3class(samples: int, a: float = 0.8, c: float = 0.33,
                         f: float = 1.0, seed: int = SEED):
    """Generalization of the previous set for three separate classes"""
    rng = np.random.default_rng(seed)
    n_per_class = samples // 3
    X, y = [], []
    count = [0, 0, 0]

    while min(count) < n_per_class:
        x = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        frontier = -a * np.sin(np.pi * f * x[0])
        if x[1] < frontier - c:
            label = 0
        elif x[1] < frontier + c:
            label = 1
        else:
            label = 2
        if count[label] < n_per_class:
            X.append(x)
            y.append(label)
            count[label] += 1

    X = np.array(X)
    y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]

def spiral_dataset(samples: int, classes: int = 4, seed: int = SEED, laps: int = 1):
    """Generalization of the previous set for three separate classes"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    count = 0

    for _ in range(samples//classes):
        for label in range(classes):
            t = rng.uniform(0.05, 1.0)
            noise = rng.normal(0.0, 0.01, size=(2,))
            theta = 2*np.pi*t*laps + 2*np.pi*label/float(classes)
            x = [t*np.cos(theta)+noise[0],t*np.sin(theta)+noise[1]]
            X.append(x)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def plot_set(dataset: SplittedDb, sinus:bool=True):

    fig, ax = plt.subplots(figsize=(16, 12))
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
                s=170 if split_name != 'train' else 50,
                edgecolors='black',
                linewidths=0.6,
                label=f'{split_name} - class {cls}'
            )

    ax.set_xlabel('x₀', fontsize=40)
    ax.set_ylabel('x₁', fontsize=40)
    if sinus:
        x_range = np.linspace(-1.0, 1.0, 300)
        ax.plot(x_range, -0.8 * np.sin(np.pi * x_range), color='black', linestyle='--',
        linewidth=2, label=f'Decision frontier: y=-{0.8}·sin(πx)', zorder=5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc='best', fontsize=30, ncol=1)
    plt.tight_layout()
    plt.savefig('split_visualization.png', dpi=150)
    plt.show()
    plt.close()
    
    

X,y = sinus_dataset_3class(samples=SAMPLES)
dataset = PairsSet(X,y)
plot_dataset(X,y,3)
labels = np.asarray(dataset.test.labels)
print(len(labels), (labels == 0).sum(), (labels == 1).sum())
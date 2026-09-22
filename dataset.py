from itertools import combinations
from sklearn.model_selection import train_test_split
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True) #enables 64 bit float numbers

SEED = 42
SAMPLES = 250

class DataBase:
    """Stores a database consisting of single input feature vectors and their respective labels.
    Includes operations that you can apply to it: 
                - Separate data into train/val/test
                - Create all possible pairs"""
    def __init__(self, X, y):
        self.features = jnp.array(X, dtype=jnp.float64)
        self.labels = jnp.array(y, dtype=jnp.float64)

    def split_raw_data(self):
        """Splits the elements into train/val/test (50/25/25), before
        any pairs are built, to avoid leaking information between subsets."""
        xs_train, xs_test, ys_train, ys_test = train_test_split(
            self.features, self.labels, test_size=0.5, random_state=SEED)
        xs_val, xs_test, ys_val, ys_test = train_test_split(
            xs_test, ys_test, test_size=0.5, random_state=SEED)
        return {'train': DataBase(xs_train, ys_train), 
                'val': DataBase(xs_val, ys_val), 
                'test': DataBase(xs_test, ys_test)}

    def build_pairs(self):
        """Builds every possible pair of elements within this subset,
        labelling each pair 1 if both elements share a class, 0 otherwise."""
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
    """Stores a dataset splitted into train/val/test"""
    def __init__(self, database: DataBase):
        sets = database.split_raw_data()
        for name, db in sets.items():
            setattr(self, name, db)


class PairsSet:
    """Stores a dataset that consists of the pairs of
    an original dataset X,y. The pairs are already splitted
    into test/val/test"""
    def __init__(self, X, y):
        self.original = DataBase(X, y)
        self.split = SplittedDb(self.original)
        for name in ('train', 'val', 'test'):
            setattr(self, name, getattr(self.split, name).build_pairs())
        self.len_train = len(self.train.labels)


def sinus_dataset_generator(samples: int, a: float = 0.8, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
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

X, y = sinus_dataset_generator(samples=SAMPLES, seed=SEED)
raw_data = DataBase(X, y)
split_data = SplittedDb(raw_data)
dataset = PairsSet(X, y)
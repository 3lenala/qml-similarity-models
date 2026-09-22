# qml-similarity-models



# Quantum architectures for similarity problems



Undergraduate thesis (Physics, EHU) — grade: 9.5/10.



Two quantum architectures for similarity problems: instead of asking

which class an input belongs to, they answer whether two inputs belong

to the same class. Both are built on top of quantum neural networks.

The idea is that if a classifier, in this case, a quantum neural

network, assigns the same class to two inputs, the states (or

measurements) it produces for them should be similar. Given two

elements, an algorithm can feed them to two copies of the same quantum

neural network and compare them with the following metrics, after all

the gates have been applied.

- **Overlap architecture**: before a measurement is taken, the

  prediction is the inner product between the two states created by 

the quantum neural network, computed through

  an inversion test.

- **Similarity function architecture** — a classical Gaussian RBF over

  the two measurements, taken after each branch is measured.

These metrics, considering both quantum neural networks, are the ones

used for training.


\## Main result



The overlap architecture cannot train at all with a single

re-uploading layer: the trainable part of the circuit cancels out by

unitarity. This is proven analytically in the thesis and verified

experimentally in this code (see `test_circuits.py` and

`test_train.py` for the corresponding tests).



## Structure



- `dataset.py`: generates the synthetic dataset (a sinusoidal decision

 boundary) and builds the train/val/test split and pairs, split before

pairing to avoid data leakage.

- `circuits.py`: the quantum circuits: encoding, variational forms

 (Two-Local, Tree Tensor), and the two architectures.

- `train.py`: training loop, early stopping, and evaluation

 (accuracy, ROC, AUC).

- `test_*.py`: unit tests, including a direct verification of the

 L=1 result above.

- `experiments.py`: Functions created to run different experiments

and graphs. 


- `executable.ipynb`: Notebook that executes the training for 

L = 1 to L = 10 data reuploading layers.


## Setup



```bash

pip install -r requirements.txt

```



## Running



```python

from dataset import sinus_dataset_generator, PairsSet

from circuits import SQNNModel, Architecture, configurations, HyperparametersModel

from train import train_model



X, y = sinus_dataset_generator(samples=250)

raw_data = DataBase(X, y)

split_data = SplittedDb(raw_data)

dataset = PairsSet(X, y)

model = SQNNModel(Architecture.OVERLAP, configurations['angle-twolocal-1'],

                 HyperparametersModel())

result = train_model(model, layers=5, dataset=dataset, seeds=10, evaluate=True)

```



## Tests



```bash 

pytest test_*.py

```


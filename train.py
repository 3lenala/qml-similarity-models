import sys
from circuits import *
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)
import time
import numpy as np
import optax
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from typing import Callable, NamedTuple

# number of repetitions of each experiment over
# different initial parameters each
SEEDS_HYPER = 40
SEEDS_EVALUATE = 10
THRESHOLD = 0.5
SCALING_FACTOR = 0.2 * jnp.pi


class TrainingResult:
    """Stores all the relevant metrics obtained in the training loop"""
    def __init__(self, theta, best_val_loss, final_train_loss, val_losses, train_losses, epochs, total_time):
        self.theta = theta
        self.epochs = epochs
        self.best_val_loss = best_val_loss
        self.time = total_time
        self.val_losses = val_losses
        self.train_losses = train_losses
        self.final_train_loss = final_train_loss
        
        self.accuracy = None
        self.roc = None
        self.auc = None
        self.confusion = None
        
    def append_eval(self, accuracy, roc, auc, confusion):
        self.accuracy = accuracy
        self.roc = roc
        self.auc = auc
        self.confusion = confusion


class SeedStorage:
    def __init__(self):
        self.val_loss = []
        self.time = []
        self.epochs = []
        
    
    def append_seed(self, training_result: TrainingResult):
        self.val_loss.append(training_result.best_val_loss)
        self.time.append(training_result.time)
        self.epochs.append(training_result.epochs)
 

class TrainedModel:
     def __init__(self, best_training: TrainingResult, over_seeds: SeedStorage):
        self.best = best_training
        
        self.val_over_seeds = over_seeds.val_loss
        self.avg_val_over_seeds = np.average(over_seeds.val_loss)
        self.std_val_over_seeds = np.std(over_seeds.val_loss)
        self.avg_time = np.average(over_seeds.time)
        self.std_time = np.std(over_seeds.time)
        self.avg_epochs = np.average(over_seeds.epochs)
        self.std_epochs = np.std(over_seeds.epochs) 

        self.acc_over_seeds = over_seeds.accuracy
        self.auc_over_seeds = over_seeds.auc
        
class TrainFunctions(NamedTuple):
    loss : Callable
    optimizer : optax.GradientTransformation
    step: Callable
        
        
def compile_train_functions(similarity_measurement: Callable, learning_rate: float):
    @jax.jit
    def loss(theta, X, y):
        """Defines the loss function as the mean squared error"""
        def single_loss(x, label):
            pred = similarity_measurement(x, theta)
            return (pred - label) ** 2

        losses = jax.vmap(single_loss, in_axes=(0, 0))(X, y)
        return jnp.mean(losses)
    
    optimizer = optax.adam(learning_rate)

    @jax.jit
    def step(params, opt_state, X, y):
        """Single gradient descent step"""
        loss_val, grads = jax.value_and_grad(loss)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    
    return TrainFunctions(loss=loss, optimizer=optimizer, step=step)
        


def compile_predict(similarity_measurement):
    @jax.jit
    def predict(X, theta):
        """Returns the set of predictions for all pairs of input vectors in X"""
        def single_predict(x):
            """Prediction over input pairs"""
            return similarity_measurement(x, theta)
        return jax.vmap(single_predict)(X)
    return predict



def train_model(model, hyperparameters_model, layers, dataset, seeds, evaluate=False) -> TrainedModel:
    """Runs the training process over a dataset for a model with layers data reuploading layers and some hyperparameters.
       In order to find the best set of parameters for the model, the training process is repeated a number SEEDS of times 
       each with a different initial value for the parameters theta of the model. Finally, the set of parameters that give a lower value
       of validation loss is kept as final set of parameters, which is then used to evaluate the model."""
       
    shape = get_theta_shape(model, layers)
    best_loss = float('inf')

    storage = SeedStorage()
    similarity_measurement = model.qnode_generator(layers)
    
    # loss, predict and step are declared here instead of in the train loop
    # to avoid the compilation process to run once per seed.
    train_functions = compile_train_functions(similarity_measurement, hyperparameters_model.learning_rate)
    
    if evaluate:
        predict = compile_predict(similarity_measurement)

    
    for seed in range(seeds):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, shape)*SCALING_FACTOR
        training_result = train(train_functions, theta0, hyperparameters_model, dataset, key)
            
        if training_result.best_val_loss < best_loss:
            if evaluate:
                training_result = evaluate_test(training_result, dataset, predict)

            best_loss = training_result.best_val_loss
            best_result = training_result
                
        storage.append_seed(training_result)

    return TrainedModel(best_result, storage)


def train(train_functions, theta0, hyperparameters_model, dataset, key, shuffle=True) -> TrainingResult:
    """Training loop: trains the model by finding the values of the parameters theta that minimize the losses of the validation set
    returns the final parameters, losses of the validation set over the epochs, the total number of epochs needed and the time
    the loop over the epochs took."""
    loss = train_functions.loss
    optimizer = train_functions.optimizer
    step = train_functions.step
    
    theta = theta0

    opt_state = optimizer.init(theta)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_theta = theta

    t_start = time.perf_counter()
    for epoch in range(hyperparameters_model.max_epochs):  # LOOP 1: over the epochs
        epoch_loss = 0.0

        key, subkey = jax.random.split(key)
        # shuffle the data so that the batches used
        # during each epoch are different
        perm = jax.random.permutation(subkey, dataset.len_train)
        X = dataset.train.features[perm]
        y = dataset.train.labels[perm]


        for start in range(0, dataset.len_train, hyperparameters_model.batch_size):  # LOOP 2: over the batches
            # for batch_size = 1: stochastic gradient descent
            # for batch_size = dataset.train.len_train: batch gradient descent
            # for any other value of batch_size in between: mini-batch gradient descent

            end = start + hyperparameters_model.batch_size
            batch_pairs = X[start:end]   # slices the training dataset into batches
            batch_labels = y[start:end]  # of size hyperparameters.batch_size
            # the loss function and the gradient will be computed over
            # one of these batches before updating the trainable parameters

            theta, opt_state, batch_loss = step(
                theta, opt_state, batch_pairs, batch_labels)

        train_loss = float(loss(theta, dataset.train.features, dataset.train.labels))
        train_losses.append(train_loss)
        val_loss = float(loss(theta, dataset.val.features, dataset.val.labels))
        val_losses.append(val_loss)
        epochs_used = epoch + 1
        
        # Early stopping check
        if val_loss < best_val_loss - hyperparameters_model.tolerance:  # continue
            best_val_loss = val_loss
            final_train_loss = train_loss
            best_theta = theta
            patience_counter = 0
            
        else:
            patience_counter += 1

        if patience_counter >= hyperparameters_model.patience:
            break

    t_end = time.perf_counter()
    epochs_used = epoch + 1
    total_time = (t_end-t_start)

    return TrainingResult(best_theta, best_val_loss, final_train_loss, val_losses, train_losses, epochs_used, total_time)


def evaluate_test(training_result: TrainingResult, dataset: PairsSet, predict: Callable, threshold: float =THRESHOLD) -> TrainingResult:

    output = predict(dataset.test.features, training_result.theta)
    y_pred = (output > threshold).astype(jnp.int64)

    accuracy = accuracy_score(y_true=dataset.test.labels, y_pred=y_pred)
    roc = roc_curve(y_true=dataset.test.labels, y_score=output)
    auc_score = roc_auc_score(y_true=dataset.test.labels, y_score=output)
    confusion = confusion_matrix(y_true=dataset.test.labels, y_pred=y_pred)
    
    training_result.append_eval(accuracy=accuracy, roc=roc, auc=auc_score, confusion=confusion)
    return training_result


def get_theta_shape(model: SQNNModel | QNNModel, layers: int) -> tuple:
    """Returns the correct shape for theta as a function of the chosen implementation

    For Tree Tensor circuits the last dimension is fixed to 2.
    Otherwise, it contains the number of rotations applied in the circuit"""
    if model.configuration.variational.form is VariationalForm.TREETENSOR:
        return (layers, model.configuration.qubits, 2)
    else:
        return (layers, model.configuration.qubits, model.configuration.variational.rotations)
    
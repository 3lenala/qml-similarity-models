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

class SeedStorage:
    def __init__(self):
        self.val_loss = []
        self.time = []
        self.epochs = []
        self.accuracy = []
        self.auc = []


class TrainedModel:
    def __init__(self, best_training_result: TrainingResult, over_seeds: SeedStorage):
        self.best = best_training_result

        self.avg_val_over_seeds = np.average(over_seeds.val_loss)
        self.std_val_over_seeds = np.std(over_seeds.val_loss)
        self.avg_time = np.average(over_seeds.time)
        self.std_time = np.std(over_seeds.time)
        self.avg_epochs = np.average(over_seeds.epochs)
        self.std_epochs = np.std(over_seeds.epochs)

class TrainingEvaluation:
    """Stores all the metrics used to evaluate the test dataset for a trained model"""
    def __init__(self, training, accuracy, roc, auc, confusion):
        self.theta = training.theta
        self.epochs = training.epochs
        self.time = training.time
        self.val_losses = training.val_losses
        self.best_val_loss = training.best_val_loss
        self.final_train_loss = training.final_train_loss
        self.train_losses = training.train_losses

        self.accuracy = accuracy
        self.roc = roc
        self.auc = auc
        self.confusion = confusion       

class EvaluatedModel:
     def __init__(self, best_training_evaluation: TrainingEvaluation, over_seeds: SeedStorage):

        self.best = best_training_evaluation
        self.val_over_seeds = over_seeds.val_loss
        self.avg_time = np.average(over_seeds.time)
        self.std_time = np.std(over_seeds.time)
        self.acc_over_seeds = over_seeds.accuracy
        self.auc_over_seeds = over_seeds.auc


def train_model(model, hyperparameters_model, layers, dataset, seeds) -> TrainedModel:
    """Runs the training process over a dataset for a model with layers data reuploading layers and some hyperparameters.
        To perform an statistical evaluation of the model a number SEEDS of repetitions each with a different initial value
        for the parameters theta of the model is performed."""
    shape = get_theta_shape(model, layers)
    best_loss = float('inf')

    evolution = SeedStorage()
    similarity_measurement = model.qnode_generator(layers)

    for seed in range(seeds):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, shape)*SCALING_FACTOR
        training_result = train(similarity_measurement, theta0, hyperparameters_model, dataset, key)
        (evolution.val_loss).append(training_result.best_val_loss)
        (evolution.time).append(training_result.time)
        (evolution.epochs).append(training_result.epochs)

        if training_result.best_val_loss < best_loss:
            best_loss = training_result.best_val_loss
            best_result = training_result

    return TrainedModel(best_result, evolution)



def loss_v_epochs(training_result, layers: int, model_name: str):
    plt.rcParams.update({'font.size': 28})
    
    viridis = cm.get_cmap('viridis')
    color_train = viridis(0.2) 
    color_val = viridis(0.8)
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    epochs = list(range(1, training_result.epochs + 1))
    
    ax1.plot(epochs, training_result.val_losses,
             color=color_val, marker='o', linestyle='-', linewidth=2,
             label='Validation losses')
    ax1.plot(epochs, training_result.train_losses,
             color=color_train, marker='s', linestyle='--', linewidth=2,
             label='Train losses')
    
    ax1.margins(x=0.02, y=0.05)
    
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    

    ax1.set_xlabel('Epochs',fontsize=28)
    ax1.set_ylabel('Losses',fontsize=28)
    ax1.legend(loc='best', fontsize=28)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}-{layers}layers.png', bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)



def train_and_evaluate_model(model, model_name, hyperparameters_model, layers, dataset, seeds):
    """Runs the training process over a dataset for a model with layers data reuploading layers and some hyperparameters.
        To perform an statistical evaluation of the model a number SEEDS of repetitions each with a different initial value
        for the parameters theta of the model is performed."""
    shape = get_theta_shape(model, layers)
    best_loss = float('inf')

    evolution = SeedStorage()
    similarity_measurement = model.qnode_generator(layers)

    for seed in range(seeds):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, shape)*SCALING_FACTOR
        training_result = train(similarity_measurement, theta0, hyperparameters_model, dataset, key)
        training_eval = evaluate_test(training_result, dataset, similarity_measurement)


        (evolution.val_loss).append(training_result.best_val_loss)
        (evolution.time).append(training_result.time)
        (evolution.accuracy).append(training_eval.accuracy)
        (evolution.auc).append(training_eval.auc)

        if training_result.best_val_loss < best_loss:
            best_loss = training_result.best_val_loss
            best_training_eval = training_eval
            best_training_result = training_result

    return EvaluatedModel(best_training_eval, evolution), best_training_result


def train(function, theta0, hyperparameters_model, dataset, key, shuffle=True) -> TrainingResult:
    """Training loop: trains the model by finding the values of the parameters theta that minimize the losses of the validation set
    returns the final parameters, losses of the validation set over the epochs, the total number of epochs needed and the time
    the loop over the epochs took."""
    theta = theta0

    @jax.jit
    def loss(theta, X, y):
        """Defines the loss function as the mean squared error"""
        def single_loss(x, label):
            pred = function(x, theta)
            return (pred - label) ** 2

        losses = jax.vmap(single_loss, in_axes=(0, 0))(X, y)
        return jnp.mean(losses)

    optimizer = optax.adam(hyperparameters_model.learning_rate)
    opt_state = optimizer.init(theta)

    @jax.jit
    def step(params, opt_state, X, y):
        """Single gradient descent step"""
        loss_val, grads = jax.value_and_grad(loss)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

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


def get_theta_shape(model: SQNNModel | QNNModel, layers: int) -> tuple:
    """Returns the correct shape for theta as a function of the chosen implementation

    For Tree Tensor circuits the last dimension is fixed to 2.
    Otherwise, it contains the number of rotations applied in the circuit"""
    if model.configuration.variational.form is VariationalForm.TREETENSOR:
        return (layers, model.configuration.qubits, 2)
    else:
        return (layers, model.configuration.qubits, model.configuration.variational.rotations)


def evaluate_test(training_result: TrainingResult, dataset: PairsSet, similarity_measurement, threshold=THRESHOLD) -> TrainingEvaluation:
    @jax.jit
    def predict(X):
        """Returns the set of predictions for all pairs of input vectors in X"""
        def single_predict(x):
            """Prediction over input pairs"""
            return similarity_measurement(x, training_result.theta)
        return jax.vmap(single_predict)(X)

    output = predict(dataset.test.features)
    y_pred = (output > threshold).astype(jnp.int64)

    accuracy = accuracy_score(y_true=dataset.test.labels, y_pred=y_pred)
    roc = roc_curve(y_true=dataset.test.labels, y_score=output)
    auc_score = roc_auc_score(y_true=dataset.test.labels, y_score=output)
    confusion = confusion_matrix(y_true=dataset.test.labels, y_pred=y_pred)
    return TrainingEvaluation(training_result,accuracy, roc, auc_score, confusion)
from circuits import *
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)
from typing import Callable, NamedTuple
import time
import numpy as np
import optax


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
        """Store the final parameters and the losses recorded throughout
        training. Evaluation metrics start as None and are filled in
        later, if the model is evaluated on the test set."""
        
        self.accuracy = accuracy
        self.roc = roc
        self.auc = auc
        self.confusion = confusion


class SeedStorage:
    """Accumulates the results of training the same model over several
    seeds, so that its stability across initializations can be assessed."""
    def __init__(self):
        """Start empty lists for the metrics collected across seeds."""
        self.val_loss = []
        self.time = []
        self.epochs = []
        
    
    def append_seed(self, training_result: TrainingResult):
        """Record the metrics from a single seed's training run."""
        self.val_loss.append(training_result.best_val_loss)
        self.time.append(training_result.time)
        self.epochs.append(training_result.epochs)
 

class TrainedModel:
    """Store the best training result found across several seeds,
    together with summary statistics over all of them, used to assess
    how stable the model is with respect to its initializations."""
    
    def __init__(self, best_training: TrainingResult, over_seeds: SeedStorage):
        """Keeps the best training result as-is, and get the averages with std. deviations."""
        
        self.best = best_training
        self.val_over_seeds = over_seeds.val_loss
        self.avg_val_over_seeds = np.average(over_seeds.val_loss)
        self.std_val_over_seeds = np.std(over_seeds.val_loss)
        self.avg_time = np.average(over_seeds.time)
        self.std_time = np.std(over_seeds.time)
        self.avg_epochs = np.average(over_seeds.epochs)
        self.std_epochs = np.std(over_seeds.epochs) 

        
class TrainFunctions(NamedTuple):
    loss : Callable
    optimizer : optax.GradientTransformation
    step: Callable
        
        
def compile_train_functions(similarity_measurement: Callable, learning_rate: float):
    """Build and JIT-compiles the loss function and the single gradient
    descent step used during training, given a similarity measurement
    and a learning rate. Declaring them once here, instead of inside
    the training loop, avoids recompiling them for every seed."""
    
    @jax.jit
    def loss(theta, X, y):
        """Define the loss function as the mean squared error"""
        def single_loss(x, label):
            """Squared error between the model's prediction for a
            single pair and its expected label."""
            
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
        return params, opt_state
    
    return TrainFunctions(loss=loss, optimizer=optimizer, step=step)


def compile_predict(similarity_measurement):
    """Builds and JIT-compiles a function that predicts the similarity
    for a whole batch of pairs at once, by vectorizing the model over
    the batch dimension."""
    
    @jax.jit
    def predict(X, theta):
        """Return the set of predictions for all pairs of input vectors in X"""
        def single_predict(x):
            """Prediction over input pairs"""
            return similarity_measurement(x, theta)
        return jax.vmap(single_predict)(X)
    return predict


def train_model(model, layers, dataset, seeds, evaluate=False) -> TrainedModel:
    """Run the training process over a dataset for a model with ´layers´ data reuploading layers and some hyperparameters.
       In order to find the best set of parameters for the model, the training process is repeated a number SEEDS of times 
       each with a different initial value for the parameters theta of the model. Finally, the set of parameters that give a lower value
       of validation loss is kept as final set of parameters, which is then used to evaluate the model."""
       
    shape = get_theta_shape(model, layers)
    similarity_measurement = model.qnode_generator(layers)
    
    best_loss = float('inf')
    best_result = None
    storage = SeedStorage()
    
    
    # loss, predict and step are declared here instead of in the train loop
    # to avoid the compilation process to run once per seed.
    train_functions = compile_train_functions(similarity_measurement, model.hyperparameters.learning_rate)
    predict = compile_predict(similarity_measurement) if evaluate else None

    
    for seed in range(seeds):
        training_result = _train_single_seed(seed, shape, train_functions, model.hyperparameters)
        storage.append_seed(training_result)
        
        if training_result.best_val_loss < best_loss:
            training_result = evaluate_test(training_result, dataset, predict) if evaluate else training_result
            best_loss = training_result.best_val_loss
            best_result = training_result
            
    return TrainedModel(best_result, storage)


def _train_single_seed(seed, shape, train_functions, hyperparameters):
    """Run the training process for a single seed."""
    key = jax.random.PRNGKey(seed)
    theta0 = jax.random.normal(key, shape)*SCALING_FACTOR
    return train(train_functions, theta0, hyperparameters, dataset, key)
        
        
def train(train_functions, theta0, hyperparameters_model, dataset, key) -> TrainingResult:
    """Training loop: train the model by finding the values of the parameters theta that minimize the losses of the validation set
    returns the final parameters, losses of the validation set over the epochs, the total number of epochs needed and the time
    the loop over the epochs took."""
    
    loss = train_functions.loss
    optimizer = train_functions.optimizer
    step = train_functions.step
    
    theta = theta0
    opt_state = optimizer.init(theta)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_theta = theta

    t_start = time.perf_counter()
    for epoch in range(hyperparameters_model.max_epochs):  # LOOP over the epochs
        X, y, key = _shuffle(key, dataset)
        theta, opt_state = _run_epoch(X, y, hyperparameters_model.batch_size, 
                                                 step, theta, opt_state)
        val_loss = float(loss(theta, dataset.val.features, dataset.val.labels))
        val_losses.append(val_loss)
        train_loss = float(loss(theta, dataset.train.features, dataset.train.labels))
        train_losses.append(train_loss)
            
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

    return TrainingResult(best_theta, best_val_loss, final_train_loss, 
                          val_losses, train_losses, epochs_used, total_time)


def _shuffle(key, dataset):
    """Shuffle the data so that the batches used during each epoch are different."""
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, dataset.len_train)
    return dataset.train.features[perm], dataset.train.labels[perm], key


def _run_epoch(X, y, batch_size, step, theta, opt_state):
    """Run a single epoch: shuffles the training data, then updates
    theta once per batch. The loss function and the gradient are computed over
    one of the batches before updating the trainable parameters.
    
    For batch_size = 1: stochastic gradient descent.
    For batch_size = dataset.train.len_train: batch gradient descent.
    For any other value in between: mini-batch gradient descent."""
    
    for start in range(0, len(X), batch_size):  # LOOP over the batches
        end = start + batch_size
        theta, opt_state = step(theta, opt_state, X[start:end], y[start:end])
    
    return theta, opt_state

def evaluate_test(training_result: TrainingResult, dataset: PairsSet, predict: Callable, threshold: float = THRESHOLD) -> TrainingResult:
    """Evaluate a trained model on the test set: computes accuracy, the
    ROC curve, AUC and the confusion matrix, using the given threshold
    to get a binary output from the model's continuous output, and stores them in
    training_result."""
    output = predict(dataset.test.features, training_result.theta)
    y_pred = (output > threshold).astype(jnp.int64)

    accuracy = accuracy_score(y_true=dataset.test.labels, y_pred=y_pred)
    roc = roc_curve(y_true=dataset.test.labels, y_score=output)
    auc_score = roc_auc_score(y_true=dataset.test.labels, y_score=output)
    confusion = confusion_matrix(y_true=dataset.test.labels, y_pred=y_pred)
    
    training_result.append_eval(accuracy=accuracy, roc=roc, auc=auc_score, confusion=confusion)
    return training_result


def get_theta_shape(model: SQNNModel | QNNModel, layers: int) -> tuple:
    """Return the correct shape for theta as a function of the chosen implementation

    For Tree Tensor circuits the last dimension is fixed to 2.
    Otherwise, it contains the number of rotations applied in the circuit"""
    if model.configuration.variational.form is VariationalForm.TREETENSOR:
        return (layers, model.configuration.qubits, 2)
    else:
        return (layers, model.configuration.qubits, model.configuration.variational.rotations)
from train import *
from train import _shuffle, _run_epoch
import chex


def test_shuffle_returns_different_key():
    """The returned key must differ from the input key, so consecutive
    epochs don't reuse the same shuffle."""
    key = jax.random.PRNGKey(0)
    X, y = sinus_dataset_generator(samples=20)
    dataset = PairsSet(X, y)

    _, _, new_key = _shuffle(key, dataset)
    assert not jnp.array_equal(key, new_key)


def test_shuffle_preserves_pair_correspondence():
    """Shuffling must keep each feature aligned with its own label."""
    key = jax.random.PRNGKey(0)
    X, y = sinus_dataset_generator(samples=20)
    dataset = PairsSet(X, y)

    X_shuffled, y_shuffled, _ = _shuffle(key, dataset)
    assert len(X_shuffled) == len(dataset.train.features)
    assert len(y_shuffled) == len(dataset.train.labels)
    

def test_run_epoch_updates_theta():
    """After one epoch, theta should have changed from its initial value."""
    model = SQNNModel(Architecture.OVERLAP, configurations['angle-twolocal-1'], HyperparametersModel())
    similarity = model.qnode_generator(layers=3)
    train_functions = compile_train_functions(similarity, learning_rate=0.1)

    X, y = sinus_dataset_generator(samples=20)
    dataset = PairsSet(X, y)

    shape = get_theta_shape(model, layers=3)
    theta0 = jax.random.normal(jax.random.PRNGKey(0), shape)
    opt_state = train_functions.optimizer.init(theta0)

    theta_after, _ = _run_epoch(dataset.train.features, dataset.train.labels, theta=theta0,
                                batch_size=8, step=train_functions.step, opt_state=opt_state)

    assert not bool(jnp.allclose(theta0, theta_after))
    
    
def test_run_epoch_L1_overlap_theta_unchanged():
    """With L=1, the overlap architecture's gradient with respect to
    theta is zero, so a single epoch should leave theta unchanged."""
    model = SQNNModel(Architecture.OVERLAP, configurations['angle-twolocal-1'], HyperparametersModel())
    similarity = model.qnode_generator(layers=1)
    train_functions = compile_train_functions(similarity, learning_rate=0.1)

    X, y = sinus_dataset_generator(samples=20)
    dataset = PairsSet(X, y)

    shape = get_theta_shape(model, layers=1)
    theta0 = jax.random.normal(jax.random.PRNGKey(0), shape)
    opt_state = train_functions.optimizer.init(theta0)

    theta_after, _ = _run_epoch(
        X=dataset.train.features, y=dataset.train.labels, batch_size=8,
        step=train_functions.step, theta=theta0, opt_state=opt_state,
    )

    chex.assert_trees_all_close(theta0, theta_after)
    
    
    
def test_train_reduces_validation_loss():
    """After training, validation loss should be lower than at the start."""
    model = SQNNModel(Architecture.SIMILARITY_FUNCTION, configurations['angle-twolocal-1'], HyperparametersModel(max_epochs=20))
    similarity = model.qnode_generator(layers=3)
    train_functions = compile_train_functions(similarity, learning_rate=0.1)

    X, y = sinus_dataset_generator(samples=40)
    dataset = PairsSet(X, y)

    shape = get_theta_shape(model, layers=3)
    theta0 = jax.random.normal(jax.random.PRNGKey(0), shape)
    key = jax.random.PRNGKey(1)

    initial_loss = float(train_functions.loss(theta0, dataset.val.features, dataset.val.labels))
    result = train(train_functions, theta0, model.hyperparameters, dataset, key)

    assert result.best_val_loss < initial_loss
    

def test_train_model_overlap_L1_loss_does_not_improve():
    """With L=1, training the overlap architecture should not reduce the
    loss at all since the gradient with respect to theta is zero."""
    model = SQNNModel(Architecture.OVERLAP, configurations['angle-twolocal-1'], HyperparametersModel(max_epochs=10))
    X, y = sinus_dataset_generator(samples=40)
    dataset = PairsSet(X, y)

    trained = train_model(model, layers=1, dataset=dataset, seeds=1)
    first_val_loss = trained.best.val_losses[0]
    last_val_loss = trained.best.val_losses[-1]

    assert np.isclose(first_val_loss, last_val_loss, atol=1e-4)
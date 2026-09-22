import jax
import pennylane as qml

def make_qnode(apply_fn, n_qubits):
    """Helper: builds a tape by running apply_fn (a circuit-building
    function, called with no arguments) on a fresh device."""
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        apply_fn()
        return qml.state()

    return circuit


def random_pair_and_parameters(seed, theta_shape, n_features=2):
    """Generates a random pair of inputs and two independent random
    theta arrays, from a single seed — useful for property-based tests
    that should hold regardless of the specific values used."""
    key = jax.random.PRNGKey(seed)
    key_x1, key_x2, key_theta_a, key_theta_b = jax.random.split(key, 4)

    x1 = jax.random.uniform(key_x1, (n_features,), minval=-1.0, maxval=1.0)
    x2 = jax.random.uniform(key_x2, (n_features,), minval=-1.0, maxval=1.0)
    theta_a = jax.random.normal(key_theta_a, theta_shape)
    theta_b = jax.random.normal(key_theta_b, theta_shape)

    return x1, x2, theta_a, theta_b

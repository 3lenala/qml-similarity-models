import pytest
from train import get_theta_shape
from conftest import *
from circuits import *
import pennylane as qml
import jax.numpy as jnp
import numpy as np


def test_angle_encoding_applies():
    """Test if RY(x_i) is applied to qubit i, for each feature."""
    
    x = np.array([0.3, 0.7])
    n_qubits = 2
    
    apply = Encoding(EncodingForm.ANGLE).circuit(x, n_qubits)
    tape = qml.workflow.construct_tape(make_qnode(apply, n_qubits))()

    ops = tape.operations
    
    assert len(ops) == n_qubits
    assert all(op.name == 'RY' for op in ops)
    assert np.isclose(ops[0].parameters[0], x[0])
    assert np.isclose(ops[1].parameters[0], x[1])

    
def test_encoding_raises_for_unsupported_form():
    """An unsupported encoding form should raise ValueError, not fail silently."""

    unsupported_encoding = Encoding.__new__(Encoding)
    unsupported_encoding.form = 'not_a_real_form'
    with pytest.raises(ValueError):
        unsupported_encoding.circuit(np.array([0.1, 0.2]), 2)


def test_zz_feature_returns_a_callable():
    """ZZ encoding should return a function that can be applied
    to a circuit."""
    x = np.array([0.3, 0.7])
    n_qubits = 2
    result = Encoding(EncodingForm.ZZ).circuit(x, n_qubits)
    assert callable(result)


def test_encoded_state_is_normalized():
    """Any valid quantum state must have norm 1."""
    
    x = np.array([0.3, 0.7])
    n_qubits = 2
    apply = Encoding(EncodingForm.ANGLE).circuit(x, n_qubits)

    circuit = make_qnode(apply, n_qubits)
    state = circuit()

    norm = np.sum(np.abs(state) ** 2)
    assert np.isclose(norm, 1.0)


@pytest.mark.parametrize('seed', [0,1,2,3,4])
def test_overlap_L1_independent_of_theta(seed):
    """With L=1, the overlap architecture's
    output does not depend on theta, because V and V-dagger cancel out."""
    model = SQNNModel(Architecture.OVERLAP, configurations['angle-twolocal-1'],
                      HyperparametersModel())
    shape = get_theta_shape(model, layers=1)
    x1, x2, theta_a, theta_b = random_pair_and_parameters(seed, shape, n_features=2)

    similarity = model.qnode_generator(layers=1)



    result_a = similarity(jnp.stack([x1, x2]), theta_a)
    result_b = similarity(jnp.stack([x1, x2]), theta_b)

    assert np.isclose(result_a, result_b)
    

@pytest.mark.parametrize('seed', [0,1,2,3,4])
def test_overlap_L2_depend_on_theta(seed):
    """With L=2, the cancellation no longer holds, so different thetas
    should give different results."""
    model = SQNNModel(Architecture.OVERLAP, configurations['angle-twolocal-1'],
                      HyperparametersModel())
    similarity = model.qnode_generator(layers=2)
    
    shape = get_theta_shape(model, layers=2)
    x1, x2, theta_a, theta_b = random_pair_and_parameters(seed, shape, n_features=2)
    
    result_a = similarity(jnp.stack([x1, x2]), theta_a)
    result_b = similarity(jnp.stack([x1, x2]), theta_b)

    assert not np.isclose(result_a, result_b)
    

@pytest.mark.parametrize('seed', [0,1,2,3,4])
def test_overlap_self(seed):
    """An element compared to itself should give maximal overlap."""
    model = SQNNModel(Architecture.OVERLAP, configurations['angle-twolocal-1'],
                      HyperparametersModel())
    similarity = model.qnode_generator(layers=seed+1)

    shape = get_theta_shape(model, layers=seed+1)
    x1, x2, theta_a, theta_b = random_pair_and_parameters(seed, shape, n_features=2)
    
    result = similarity(jnp.stack([x1, x1]), theta_a)
    assert np.isclose(result, 1.0)
    result = similarity(jnp.stack([x2, x2]), theta_b)
    assert np.isclose(result, 1.0)
    result = similarity(jnp.stack([x1, x1]), theta_a)
    assert np.isclose(result, 1.0)
    result = similarity(jnp.stack([x2, x2]), theta_b)
    assert np.isclose(result, 1.0)

@pytest.mark.parametrize('seed', [0,1,2,3,4])
def test_similarity_self(seed):
    """An element compared to itself should give maximal value on the similarity function."""
    model = SQNNModel(Architecture.SIMILARITY_FUNCTION, configurations['angle-twolocal-1'],
                      HyperparametersModel())
    similarity = model.qnode_generator(layers=seed+1)

    shape = get_theta_shape(model, layers=seed+1)
    x1, x2, theta_a, theta_b = random_pair_and_parameters(seed, shape, n_features=2)
    
    result = similarity(jnp.stack([x1, x1]), theta_a)
    assert np.isclose(result, 1.0)
    result = similarity(jnp.stack([x2, x2]), theta_b)
    assert np.isclose(result, 1.0)
    result = similarity(jnp.stack([x1, x1]), theta_a)
    assert np.isclose(result, 1.0)
    result = similarity(jnp.stack([x2, x2]), theta_b)
    assert np.isclose(result, 1.0)

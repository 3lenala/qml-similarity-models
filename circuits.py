from enum import Enum, auto
from dataset import *
import pennylane as qml
from itertools import combinations
from dataclasses import dataclass

LAYERS_LIST = [i+1 for i in range(10)]
N_QUBITS = 2


@dataclass
class HyperparametersModel:
    """Hyperparameters selection for training:
                    - max epochs: maximum number of training epochs
                    - patience: epochs without an improvement before early stopping
                    - tolerance: minimum loss improvement to reset patience counter
                    - batch size: size of the batches created in the training
                    - learning rate: learning rate for the ADAM optimizer"""

    def __init__(
        self, max_epochs=500, patience=10, tolerance=1e-4, batch_size=32, learning_rate=0.01
    ):
        """Set the hyperparameters to their given (or default) values."""
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.learning_rate = learning_rate

# ENUMERATION OF ALL POSSIBLE CONFIGURATIONS AND ARCHITECTURES:
class Architecture(Enum):
    OVERLAP = auto()
    SIMILARITY_FUNCTION = auto()


class EntanglementForm(Enum):
    CIRCULAR = auto()
    LINEAR = auto()
    NONE = auto()


class VariationalForm(Enum):
    TWOLOCAL = auto()
    TWOLOCALVARIATION = auto()
    TREETENSOR = auto()


class EncodingForm(Enum):
    ANGLE = auto()
    ZZ = auto()



class Encoding:
    def __init__(self, form: EncodingForm):
        self.form = form
    def circuit(self, x, n_qubits):
        """Return a function that applies the encoding to a quantum circuit.
        This method has to be called for each input vector."""
        dispatch = {
            EncodingForm.ANGLE: self._angle_encoding,
            EncodingForm.ZZ: self._zz_feature
        }
        if (method := dispatch.get(self.form)) is not None:
            return method(x, n_qubits)
        raise ValueError(f'{self.form} encoding not supported')


    def _angle_encoding(self, x, n_qubits):
        """Angle encoding: RY(x_i) applied to each qubit."""
        def apply():
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
        return apply

    def _zz_feature(self, x, n_qubits, rescale = True):
        """ZZ feature map (Havlicek et al., 2019),
            by default the inputs are rescaled to [0,pi]"""
        def apply():
            x_scaled = (x + 1) * (np.pi / 2.0) if rescale else x
            nload = min(len(x_scaled), n_qubits)

            for i in range(nload):
                qml.Hadamard(i)
                qml.RZ(2 * x_scaled[i], wires=i)

            for q0, q1 in combinations(range(nload), 2):
                qml.CNOT(wires=[q0, q1])
                qml.RZ(2.0 * (np.pi - x_scaled[q1]) * (np.pi - x_scaled[q0]), wires=q1)
                qml.CNOT(wires=[q0, q1])
        return apply
    

class Variational:
    def __init__(self, entanglement: EntanglementForm, form: VariationalForm, n_rotations: int):
        self.entanglement = entanglement
        self.form = form
        self.rotations = n_rotations

    def entanglement_block(self, n_qubits):
        """Return a function that applies the chosen entanglement pattern to a quantum circuit"""
        dispatch = {
            EntanglementForm.CIRCULAR: self._circular_entanglement,
            EntanglementForm.LINEAR: self._linear_entanglement,
            EntanglementForm.NONE: self._no_entanglement
        }
        if (method := dispatch.get(self.entanglement)) is not None:
            return method(n_qubits)
        raise ValueError(f'Unknown entanglement {self.entanglement}')
    
    def _circular_entanglement(self, n_qubits):
        def apply():
            """Circular entanglement: CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1),CNOT(n-1,0)"""
            if n_qubits <= 1:
                return
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        return apply
    
    def _linear_entanglement(self, n_qubits):
        """Linear entanglement: CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1)."""
        def apply():
            if n_qubits <= 1:
                return
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])      
        return apply

    def _no_entanglement(self, n_qubits):
        """No entanglement is applied."""
        def apply():
            return
        return apply

    def circuit(self, layers, theta, n_qubits):
        """Return a function that applies the variational block to a quantum circuit.
        It takes the layers index which indicates the data reuploading layer this block belongs
        to as input. Theta is an array containing the trainable parameters"""
        entanglement_circuit = self.entanglement_block(n_qubits)
        dispatch = {
            VariationalForm.TWOLOCALVARIATION: self._two_local_variation,
            VariationalForm.TWOLOCAL: self._two_local,
            VariationalForm.TREETENSOR: self._tree_tensor
        }
        
        if (method := dispatch.get(self.form)) is not None:
            return method(layers, theta, n_qubits, entanglement_circuit)
        raise ValueError(f'Unknown variational form {self.form}')

    def _two_local_variation(self, layers, theta, n_qubits, entanglement_circuit):
        """n_rotations are applied alterning between RY/RZ followed
        by a single entanglement block at the end. Theta is of shape
        [layers, n_qubits, n_rotations]"""
        def apply():
            for rotation in range(self.rotations):
                for i in range(n_qubits):
                    if rotation % 2 == 0:
                        qml.RY(theta[layers, i, rotation], wires=i)
                    else:
                        qml.RZ(theta[layers, i, rotation], wires=i)
            entanglement_circuit()
        return apply


    def _two_local(self, layers, theta, n_qubits, entanglement_circuit):
        """Standard Two Local circuit: for each rotation step, RY rotations are applied to all qubits
        followed immediately by the entangling block.
        Theta should be of size [layers, n_qubits, n_rotations]"""
        def apply():
            for rotation in range(self.rotations):
                for i in range(n_qubits):
                    qml.RY(theta[layers, i, rotation], wires=i)
            entanglement_circuit()
        return apply


    def _tree_tensor(self, layers, theta, n_qubits, entanglement_circuit):
        """Tree Tensor form applied strictly for 2 qubits"""
        def apply():
            if n_qubits != 2:
                raise ValueError(f'Tree Tensor only available for 2 qubits')
            for i in range(n_qubits):
                qml.RY(theta[layers, i, 0], wires=i)
            qml.CNOT(wires=[1, 0])  # control=1, target=0
            qml.RY(theta[layers, 0, 1], wires=0)
        return apply


class Configuration:
    """Describe the model: number of qubits, encoding used, variational form used.
    It does not include the number of data reuploading layers since some experiments
    sweep through different possible data reuploading layers for the model."""

    def __init__(self, n_qubits, encoding, variational):
        """Store the number of qubits, the encoding and the variational
        form that together define a model's architecture."""
        self.qubits = n_qubits
        self.encoding = encoding
        self.variational = variational


class SQNNModel:
    """Store all the information required to build an overlap and a similarity 
    function architecture for a similarity problem"""
    def __init__(self, architecture: Architecture, configuration: Configuration, hyperparameters: HyperparametersModel):
        """Build a quantum device for the given configuration and prepares
        a dictionary to store the training results obtained for each
        number of data re-uploading layers."""
        self.architecture = architecture
        self.configuration = configuration
        self.dev = qml.device('default.qubit', wires=self.configuration.qubits)
        
        # to keep track of training results for different
        # number of data reuploading layers applied.
        self.result = {L: [] for L in LAYERS_LIST}
        self.hyperparameters = hyperparameters
        

    def create_qnn(self):
        """Build the standard quantum neural network as a callable function."""
        def qnn(x, theta, layers):
            """The circuit performs the encoding block followed by the variational
            block a number layers of times for layers the number of data reuploading repetitions"""
            for layer in range(layers):
                self.configuration.encoding.circuit(x, self.configuration.qubits)()
                self.configuration.variational.circuit(layer, theta, self.configuration.qubits)()
        return qnn

    def qnode_generator(self, layers):
        """Build the callable function for the similarity measurement
        This similarity measurement will be different for each of the architectures
        that will estimate the similarity between two inputs x1 and x2 in different ways:
        - Similarity function architecture: Applies the quantum neural network over the two inputs
                                            and calculates the similarity as a classical  metric between
                                            the outputs
        - Overlap architecture: Applies the quantum neural network to the inputs x1 and x2 but instead of measuring
                                to obtain an output, the inner product between the two states created is given as the output """
        qnn = self.create_qnn()
        dispatch = {
            Architecture.OVERLAP: self._overlap_squared,
            Architecture.SIMILARITY_FUNCTION: self._similarity_function
        }
        
        if (measure := dispatch.get(self.architecture)) is None:
            raise ValueError(f'Unknown architecture {self.architecture}')

        @jax.jit
        def similarity_measurement(x, theta):
            """Returns the similarity between the two elements of pair x,
            using whichever architecture this model was built with."""
            return measure(qnn, layers, x[0], x[1], theta)
        return similarity_measurement
    
    
    def _overlap_squared(self, qnn, layers, x1, x2, theta):
        """Returns |<ψ(x2)|ψ(x1)>|^2, the squared overlap
        between the two states produced by the shared-parameter
        QNN, computed via the inversion test."""
        
        @qml.qnode(self.dev, interface='jax', diff_method='backprop')
        def overlap_circuit(x1, x2, theta):
            qnn(x1, theta, layers)  # creates U(x1,theta)|0>
            qml.adjoint(qnn)(x2, theta, layers)  # adds U*(x2,theta) to U(x1,theta)|0>
            return qml.probs(wires=range(self.configuration.qubits))
        return overlap_circuit(x1, x2, theta)[0] 

    def _similarity_function(self, qnn, layers, x1, x2, theta):
        """Returns a Gaussian RBF over the two measurements
        obtained from the QNN, close to 1 when they are similar
        and close to 0 when they are far apart."""
        
        @qml.qnode(self.dev, interface='jax', diff_method='backprop')
        def qnn_ev(x, theta):
            qnn(x, theta, layers)
            return qml.expval(qml.PauliZ(0))
        pred1 = qnn_ev(x1, theta)  # evaluates U(x1, theta)|0>
        pred2 = qnn_ev(x2, theta)  # evaluates U(x2, theta)|0>
        return jnp.exp(-(pred1-pred2)**2)


class QNNModel:
    """Stores all the information required to build a standard quantum neural network
    for a classification problem"""
    def __init__(self, configuration: Configuration, hyperparameters: HyperparametersModel):
        """Builds a quantum device for the given configuration and prepares
        a dictionary to store the training results obtained for each
        number of data re-uploading layers."""
        self.configuration = configuration
        self.dev = qml.device('default.qubit', wires=self.configuration.qubits)
        # to keep track of training results for different
        # number of data reuploading layers applied.
        self.result = {L: [] for L in LAYERS_LIST}
        
        self.hyperparameters = hyperparameters


    def create_qnn(self):
        """Build the standard quantum neural network as a callable function"""
        def qnn(x, theta, layers):
            """The circuit performs the encoding block followed by the variational
            block a number layers of times for layers the number of data reuploading repetitions"""
            for layer in range(layers):
                self.configuration.encoding.circuit(x, self.configuration.qubits)()
                self.configuration.variational.circuit(layer, theta, self.configuration.qubits)()
        return qnn

    def qnode_generator(self, layers):
        """Returns a function that evaluates the expected value of PauliZ
        on the first qubit for a single input, used to classify it against
        a threshold."""
        qnn = self.create_qnn()
        @jax.jit
        def prediction(x, theta):
            """Evaluates the QNN on a single input x."""
            @qml.qnode(self.dev, interface='jax', diff_method='backprop')
            def qnn_ev(x, theta):
                qnn(x, theta, layers)
                return qml.expval(qml.PauliZ(0))
            return qnn_ev(x, theta)[0]
        return prediction
    

    def similarity_generator(self, layers):
        """Returns a function that predicts whether two inputs belong
        to the same class, using this standard QNN as baseline."""
        qnn = self.qnode_generator(layers)
        def similarity_measurement(x,theta):
            """Prediction over input pairs"""
            x1 = x[0]
            x2 = x[1]
            cls1 = (qnn(x1, theta)>0)
            cls2 = (qnn(x2, theta)>0)
            return (cls1 == cls2)
        return similarity_measurement

# Predefined model configuration used in the experiments
# The naming procedure is: {encoding}-{variational form}-
#                          {number of rotations}-{entanglement}

configurations = {
    'angle-twolocal-1': Configuration(
        n_qubits=N_QUBITS,
        encoding=Encoding(EncodingForm.ANGLE),
        variational=Variational(entanglement=EntanglementForm.LINEAR,
                                form=VariationalForm.TWOLOCAL, 
                                n_rotations=1)),

    #(n_rotations/entanglement does not apply to TreeTensor)
    'angle-treetensor': Configuration(
        n_qubits=N_QUBITS,
        encoding=Encoding(EncodingForm.ANGLE),
        variational=Variational(entanglement=EntanglementForm.NONE,
                                form=VariationalForm.TREETENSOR, 
                                n_rotations=2)),

}


models_sim = {i : SQNNModel(architecture=Architecture.SIMILARITY_FUNCTION, configuration=configurations[i], 
                            hyperparameters=HyperparametersModel()) for i in configurations}

models_overlap = {i : SQNNModel(architecture=Architecture.OVERLAP, configuration=configurations[i], 
                            hyperparameters=HyperparametersModel()) for i in configurations}

models_overlap['angle-twolocal-1'].hyperparameters = HyperparametersModel(patience=10, tolerance=1e-4, learning_rate=0.001, batch_size=16)
models_overlap['angle-treetensor'].hyperparameters = HyperparametersModel(patience=10, tolerance=1e-4, learning_rate=0.001, batch_size=8)

models_sim['angle-twolocal-1'].hyperparameters = HyperparametersModel(patience=10, tolerance=1e-4, learning_rate=0.001, batch_size=16)
models_sim['angle-treetensor'].hyperparameters = HyperparametersModel(patience=10, tolerance=1e-4, learning_rate=0.001, batch_size=8)
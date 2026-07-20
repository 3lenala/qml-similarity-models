from enum import Enum, auto
from dataset import *
import pennylane as qml
from itertools import combinations
from dataclasses import dataclass

LAYERS_LIST = [i+1 for i in range(10)]
N_QUBITS = 2


@dataclass # included so that the base values are modifyable
class HyperparametersModel:
    """Hyperparameters selection for training:
                    - max epochs: maximum number of training epochs
                    - patience: epochs without an improvement before early stopping
                    - tolerance: minimum loss improvement to reset patience counter
                    - batch size: size of the batches created in the training
                    - learning rate: learning rate for the ADAM optimizer"""

    def __init__(
        self, max_epochs, patience, tolerance, batch_size, learning_rate
    ):
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.learning_rate = learning_rate

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
        """Returns a function that applies the encoding to a quantum circuit.
        This method has to be called for each input vector"""
        if self.form is EncodingForm.ANGLE:
            def angle_encoding():
                """Angle encoding: RY(x_i) over each qubit, with RZ(x0*x1) as an
                optional addition if cros_term = True"""
                for i in range(n_qubits):
                    qml.RY(x[i], wires=i)
            return angle_encoding
        
        elif self.form is EncodingForm.ZZ:
            def zz_feature(rescale=True):
                """ZZ feature map (Havlicek et al., 2019),
                  by default the inputs are reescaled to [0,pi]"""
                x_scaled = (x + 1) * (np.pi / 2.0) if rescale else x
                nload = min(len(x_scaled), n_qubits)

                for i in range(nload):
                    qml.Hadamard(i)
                    qml.RZ(2 * x_scaled[i], wires=i)

                for q0, q1 in combinations(range(nload), 2):
                    qml.CNOT(wires=[q0, q1])
                    qml.RZ(2.0 * (np.pi - x_scaled[q1]) * (np.pi - x_scaled[q0]), wires=q1)
                    qml.CNOT(wires=[q0, q1])
            return zz_feature
        else:
            raise ValueError(f'{self.form} encoding not supported')


class Variational:
    def __init__(self, entanglement: EntanglementForm, form: VariationalForm, n_rotations: int):
        self.entanglement = entanglement
        self.form = form
        self.rotations = n_rotations

    def entanglement_block(self, n_qubits):
        """Returns a function that applies the chosen entanglement pattern to a quantum circuit"""
        if self.entanglement is EntanglementForm.CIRCULAR:
            def circular_entanglement():
                """Circular entanglement: CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1),CNOT(n-1,0)"""
                if n_qubits <= 1:
                    return
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return circular_entanglement
        elif self.entanglement is EntanglementForm.LINEAR:
            def linear_entanglement():
                """Linear entanglement: CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1)"""
                if n_qubits <= 1:
                    return
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return linear_entanglement
        elif self.entanglement is EntanglementForm.NONE:
            def no_entanglement():
                return
            return no_entanglement
        else:
            raise ValueError(f'Unknown entanglement {self.entanglement}')

    def circuit(self, layers, theta, n_qubits):
        """Returns a function that applies the variational block to a quantum circuit.
        It takes the layers index which indicates the data reuploading layer this block belongs
        to as input. Theta is an array containing the trainable parameteres"""
        entanglement_circuit = self.entanglement_block(n_qubits)
        if self.form is VariationalForm.TWOLOCALVARIATION:
            def TwoLocalVariation():
                """n_rotations are applied alterning between RY/RZ followed
                by a single entanglement block at the end. Theta is of shape
                [layers, n_qubits, n_rotations]"""
                for rotation in range(self.rotations):
                    for i in range(n_qubits):
                        if rotation % 2 == 0:
                            qml.RY(theta[layers, i, rotation], wires=i)
                        else:
                            qml.RZ(theta[layers, i, rotation], wires=i)
                entanglement_circuit()
            return TwoLocalVariation

        elif self.form is VariationalForm.TWOLOCAL:
            def TwoLocal():
                """Standard Two Local circuit: for each rotation step, RY rotations are applied to all qubits
                followed immediately by the entangling block.
                Theta should be of size [layers, n_qubits, n_rotations]"""
                for rotation in range(self.rotations):
                    for i in range(n_qubits):
                        qml.RY(theta[layers, i, rotation], wires=i)
                    entanglement_circuit()
            return TwoLocal

        elif self.form is VariationalForm.TREETENSOR:
            def TreeTensor():
                """Tree Tensor form applied strictly for 2 qubits"""
                if n_qubits != 2:
                    raise ValueError(f'Tree Tensor only available for 2 qubits')
                for i in range(n_qubits):
                    qml.RY(theta[layers, i, 0], wires=i)
                qml.CNOT(wires=[1, 0])  # control=1, target=0
                qml.RY(theta[layers, 0, 1], wires=0)
            return TreeTensor
        else:
            raise ValueError(f'Unknown variational form {self.form}')


class Configuration:
    """Describes the model: number of qubits, encoding used, variational form used.
    It does not include the number of data reuploading layers since some experiments
    sweep through different possible data reuploading layers for the model."""

    def __init__(self, n_qubits, encoding, variational):
        self.qubits = n_qubits
        self.encoding = encoding
        self.variational = variational


class SQNNModel:
    """Stores all the information required to build an overlap and a similarity 
    function architecture for a similarity problem"""
    def __init__(self, architecture: Architecture, configuration: Configuration):
        self.architecture = architecture
        self.configuration = configuration
        self.dev = qml.device('default.qubit', wires=self.configuration.qubits)
        # to keep track of training results for different
        self.result = {L: [] for L in LAYERS_LIST}
        # number of data reuploading layers applied.

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
        """Builds the callable function for the similarity measurement
        This similarity measurement will be different for each of the architectures
        that will estimate the similarity between two inputs x1 and x2 in different ways:
        - Similarity function architecture: Applies the quantum neural network over the two inputs
                                            and calculates the similarity as a classical  metric between
                                            the outputs
        - Overlap architecture: Applies the quantum neural network to the inputs x1 and x2 but instead of measuring
                                to obtain an output, the inner product between the two states created is given as the output """
        qnn = self.create_qnn()

        @jax.jit
        def similarity_measurement(x, theta):
            x1 = x[0]
            x2 = x[1]
            if self.architecture is Architecture.OVERLAP:
                def overlap_squared(x1, x2, theta):
                    @qml.qnode(self.dev, interface='jax', diff_method='backprop')
                    def overlap_circuit(x1, x2, theta):
                        qnn(x1, theta, layers)  # creates U(x1,theta)|0>
                        qml.adjoint(qnn)(x2, theta, layers)  # adds U*(x2,theta) to U(x1,theta)|0>
                        return qml.probs(wires=range(self.configuration.qubits))
                    return overlap_circuit(x1, x2, theta)[0]  # returns |<ψ(x2)|ψ(x1)>|^2.
                return overlap_squared(x1, x2, theta)  # returns a function that
                # evaluates |<ψ(x2)|ψ(x1)>|^2.

            elif self.architecture is Architecture.SIMILARITY_FUNCTION:
                def similarity_function(x1, x2, theta):
                    @qml.qnode(self.dev, interface='jax', diff_method='backprop')
                    def qnn_ev(x, theta):
                        qnn(x, theta, layers)
                        return qml.expval(qml.PauliZ(0))
                    pred1 = qnn_ev(x1, theta)  # evaluates U(x1, theta)|0>
                    pred2 = qnn_ev(x2, theta)  # evaluates U(x2, theta)|0>
                    return jnp.exp(-(pred1-pred2)**2)  # measures how far apart from each other
                    # the outputs are
                return similarity_function(x1, x2, theta)
            else:
                raise ValueError(f'Unkown architecture {self.architecture}')
        return similarity_measurement  # returns a function that
        # evaluates exp(-(QNN(x1)-QNN(x2))**2


class QNNModel:
    """Stores all the information required to build a standard quantum neural network
    for a classification problem"""
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.dev = qml.device('default.qubit', wires=self.configuration.qubits)
        # to keep track of training results for different
        self.result = {L: [] for L in LAYERS_LIST}
        # number of data reuploading layers applied.

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
        qnn = self.create_qnn()
        @jax.jit
        def prediction(x, theta):
            @qml.qnode(self.dev, interface='jax', diff_method='backprop')
            def qnn_ev(x, theta):
                qnn(x, theta, layers)
                return qml.expval(qml.PauliZ(0))
            return qnn_ev(x, theta)[0]
        return prediction
    

    def similarity_generator(self, layers):
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
        variational=Variational(entanglement=EntanglementForm.CIRCULAR,
                                form=VariationalForm.TWOLOCAL, 
                                n_rotations=1)),

    'angle-twolocal-2': Configuration(
        n_qubits=N_QUBITS,
        encoding=Encoding(EncodingForm.ANGLE),
        variational=Variational(entanglement=EntanglementForm.CIRCULAR,
                                form=VariationalForm.TWOLOCAL, 
                                n_rotations=2)),

    #(n_rotations/entanglement does not apply to TreeTensor)
    'angle-treetensor': Configuration(
        n_qubits=N_QUBITS,
        encoding=Encoding(EncodingForm.ANGLE),
        variational=Variational(entanglement=EntanglementForm.NONE,
                                form=VariationalForm.TREETENSOR, 
                                n_rotations=2)),

    'zz-treetensor': Configuration(
        n_qubits=N_QUBITS,
        encoding=Encoding(EncodingForm.ZZ),
        variational=Variational(entanglement=EntanglementForm.NONE,
                                form=VariationalForm.TREETENSOR, 
                                n_rotations=2)),
}


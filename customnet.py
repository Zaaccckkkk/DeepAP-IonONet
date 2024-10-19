from abc import ABC, abstractmethod
from itertools import cycle
from deepxde.nn import FNN
from deepxde.nn import NN
from deepxde.nn import activations
from deepxde import config
from deepxde.backend import tf


class DeepONetStrategy(ABC):
    """DeepONet building strategy.

    See the section 3.1.6. in
    L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. Karniadakis.
    A comprehensive and fair comparison of two neural operators
    (with practical extensions) based on FAIR data.
    Computer Methods in Applied Mechanics and Engineering, 393, 114778, 2022.
    """

    def __init__(self, net):
        self.net = net

    @abstractmethod
    def build(self, layer_sizes_branch, layer_sizes_trunk):
        """Build branch and trunk nets."""

    @abstractmethod
    def call(self, x_func, x_loc, training=False):
        """Forward pass."""


class SingleOutputStrategy(DeepONetStrategy):
    """Single output build strategy is the standard build method."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        branch = self.net.build_branch_net(layer_sizes_branch)
        trunk = self.net.build_trunk_net(layer_sizes_trunk)
        return branch, trunk

    def call(self, x_func, x_loc, training=False):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = self.net.merge_branch_trunk(x_func, x_loc)
        return x


class SplitBothStrategy(DeepONetStrategy):
    """Split the outputs of both the branch net and the trunk net into n groups,
    and then the kth group outputs the kth solution.

    For example, if n = 2 and both the branch and trunk nets have 100 output neurons,
    then the dot product between the first 50 neurons of
    the branch and trunk nets generates the first function,
    and the remaining 50 neurons generate the second function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        single_output_strategy = SingleOutputStrategy(self.net)
        return single_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)

    def call(self, x_func, x_loc, training=False):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        identifier = self.net.identifier_layer(x_func)  # Extract 3 neurons for classification (840, 3)
        param_value = self.net.param_value_layer(x_func)  # (840, 1)
        param_value = (
            identifier[:, 0:1] * tf.zeros_like(param_value) +  # Case 1: V1/2 and C, output 0
            identifier[:, 1:2] * (self.net.g_na_range[0] + (self.net.g_na_range[1] - self.net.g_na_range[0]) * tf.sigmoid(param_value)) +  # Case 2: G_Na, map to [84, 156]
            identifier[:, 2:3] * (self.net.g_k_range[0] + (self.net.g_k_range[1] - self.net.g_k_range[0]) * tf.sigmoid(param_value))  # Case 3: G_K, map to [30, 42]
        )  # (840, 1)

        # Split x_func and x_loc into respective outputs
        shift = 0
        size = (x_func.shape[1]) // self.net.num_outputs  # last layer has 30 neurons
        xs = []
        for _ in range(self.net.num_outputs):
            x_func_ = x_func[:, shift: shift + size]  # (840, 10)
            x_loc_ = x_loc[:, shift: shift + size]  # (4000, 10)
            x = self.net.merge_branch_trunk(x_func_, x_loc_)  # (840, 4000) for training data
            xs.append(x)
            shift += size

        xs = tf.stack(xs, axis=2)  # Stack along a new dimension (shape: [batch_size, num_outputs, feature_size]) (840, 4000, 3)
        additional_outputs = tf.concat([identifier, param_value], axis=-1)  # (840, 4)
        additional_outputs = tf.tile(tf.expand_dims(additional_outputs, axis=1), [1, x_loc.shape[0], 1])  # (840, 4000, 4)

        final_outputs = tf.concat([xs, additional_outputs], axis=2)  # (840, 4000, 7)

        return final_outputs


class CumtomizedCartesianProd(NN):

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        num_outputs=3,
        multi_output_strategy="split_both",
        regularization=None,
        activation_classifier=tf.keras.activations.softmax,
        g_na_range=(84, 156),
        g_k_range=(30, 42)
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.activation_classifier = activation_classifier

        self.num_outputs = num_outputs
        self.multi_output_strategy = {"split_both": SplitBothStrategy, }[multi_output_strategy](self)
        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )
        self.g_na_range = g_na_range
        self.g_k_range = g_k_range

        self.b = cycle(
            [
                tf.Variable(tf.zeros(1, dtype=config.real(tf)))
                for _ in range(self.num_outputs)
            ]
        )
        # Layer for outputting 3 indicators
        self.identifier_layer = tf.keras.layers.Dense(3, activation=self.activation_classifier)

        # Layer for outputting the value of the varying parameter (specific to G_Na or G_K)
        self.param_value_layer = tf.keras.layers.Dense(1, activation='linear')  # New head for parameter values

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
            # Fully connected network
        return FNN(
            layer_sizes_branch,
            self.activation_branch,
            self.kernel_initializer,
            regularization=self.regularization,
        )

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(
            layer_sizes_trunk,
            self.activation_trunk,
            self.kernel_initializer,
            regularization=self.regularization,
        )

    def merge_branch_trunk(self, x_func, x_loc):
        y = tf.einsum("bi,ni->bn", x_func, x_loc)
        y += next(self.b)
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return tf.stack(ys, axis=2)

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc, training)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        # Set the .output attribute
        self.outputs = x
        return x

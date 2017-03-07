from numpy import exp, array, random, dot
import numpy as np

class NeuralNetwork():
    """docstring for ."""

    def __init__(self):
        # seed the random number generation, so it generates the same number
        # every time the program runs
        random.seed(1)

        # we model a single neuron, with 3 input connections and 1 output connection.
        # we assign random weights to 3 x 1 matrix, with a value in the range
        # of -1 to 1 a mean of 0
        # our neuron have 3 input connections and 1 output connections

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # Activation Function
    # The sigmoid function, which describes an s shaped curve
    # we pass the weighted sum of the inputs through this function
    # to normalise them between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # gradient of the sigmoid curve
    def __sigmoid_deravative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # pass the training set through our neural net
            output = self.predict(training_set_inputs)

            # calculate the error
            error = (training_set_outputs - output)
            # multiply the error by the input ad again by the gradient of the
            # sigmoid curve
            adjustments = dot(training_set_inputs.T, error *
                              self.__sigmoid_deravative(output))

            # adjust the weights
            self.synaptic_weights += adjustments

    def predict(self, inputs):
        # pass inputs through our neural network (our singe neuron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    # initialise a single neuron neural network
    neural_network = NeuralNetwork()
    print('Random starting synaptic weights: ')
    print(neural_network.synaptic_weights)

    # the training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # train the neural network using the training sets
    # Do it 10,000 times amd making small adjustments each time

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic weights after training: ')
    print(neural_network.synaptic_weights)

    # Test the neural network
    print('predicting...')
    print('considering new situation [1, 0, 0] -> ?: ')
    print(neural_network.predict(array([1, 0, 0])))

from numpy import exp, array, random, dot
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


class NeuralNetwork():
    """docstring for NeuralNetwork."""
    def __init__(self):
        # seed the random number generation, so it generates the same number
        # every time
        random.seed(1)

        # setting the number of nodes in layer 2 and layer 3
        # more the number of nodes in hidden layer more the confident the
        # neural network in predicting
        l2 = 5
        l3 = 4

        # assign random weights to the matrix
        # (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.synaptic_weights1 = 2 * np.random.random((3, l2)) - 1
        self.synaptic_weights2 = 2 * np.random.random((l2, l3)) - 1
        self.synaptic_weights3 = 2 * np.random.random((l3, 1)) - 1

    def __sigmoid(self, x):
        return 1/(1 + exp(-x))

    def __sigmoid_deravative(self, x):
        return x * (1-x)

    # train neural network, adjusting synaptics weight each time
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            # passing training set through our neual network
            # a2 means activation fed to the second layer
            a2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            # print("Dot hota ya hai")
            # print(dot(training_set_inputs, self.synaptic_weights1))
            # print("sigmoid ka kya result hai?")
            # print(a2)
            a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
            output = self.__sigmoid(dot(a3, self.synaptic_weights3))

            # calculatng error
            delta4 = (training_set_outputs - output)
            # delta4 = (training_set_outputs - output)* self.__sigmoid_deravative(output)

            # finding errors in each layer
            delta3 = dot(self.synaptic_weights3, delta4.T) * (self.__sigmoid_deravative(a3).T)
            delta2 = dot(self.synaptic_weights2, delta3) * (self.__sigmoid_deravative(a2).T)

            # adjustments(gradient) in each layer
            adjustment3 = dot(a3.T, delta4)
            adjustment2 = dot(a2.T, delta3.T)
            adjustment1 = dot(training_set_inputs.T, delta2.T)

            # adjusting the weight accordingly
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3


    def forward_pass(self, inputs):
        # passing our inputs into neural networks
        a2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
        output = self.__sigmoid(dot(a3, self.synaptic_weights3))
        return output


if __name__ == '__main__':
    # initializing neural network
    neural_network = NeuralNetwork()

    # printing the initial weight assigned to the neural network
    print("Random starting synaptics weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("Random starting synaptics weights (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("Random starting synaptics weights (layer 3): ")
    print(neural_network.synaptic_weights3)

    # loading the training set
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    plt.imshow(training_set_inputs)
    plt.show()
    # Interpolation calculates what the color or value of a pixel “should” be,
    # according to different mathematical schemes. One common place that this
    # happens is when you resize an image. The number of pixels change, but you
    # want the same information. Since pixels are discrete, there’s missing
    # space. Interpolation is how you fill that space.
    plt.imshow(training_set_inputs, interpolation='nearest')
    plt.show()
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # training the neural network using the  training sets
    # Doing 10,000 times and making small adjustments every time

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("\nnew synaptic weights (layer 1) after training: ")
    print(neural_network.synaptic_weights1)
    print("\nnew synaptic weights (layer 2) after training: ")
    print(neural_network.synaptic_weights2)
    print("\nnew synaptic weights (layer 3) after training: ")
    print(neural_network.synaptic_weights3)

    # Testing the neural network
    print("Predicting...")
    print("considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.forward_pass(array([1, 0, 0])))

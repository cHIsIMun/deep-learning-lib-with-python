import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_nn(nn):
    G = nx.DiGraph()
    pos = {}
    x = 0
    y = nn.neurons_input/2
    # add input layer
    for i in range(nn.neurons_input):
        G.add_node(f"input_{i}", layer="input", neurons=nn.neurons_input)
        pos[f"input_{i}"] = (x, y)
        y -= 1

    x += 1
    y = nn.neurons_hidden_layers[0]/2
    # add hidden layers
    for i, neurons in enumerate(nn.neurons_hidden_layers):
        for j in range(neurons):
            G.add_node(f"hidden_{i+1}_{j}", layer="hidden", neurons=neurons)
            pos[f"hidden_{i+1}_{j}"] = (x, y)
            y -= 1
            if i == 0:
                for k in range(nn.neurons_input):
                    G.add_edge(f"input_{k}", f"hidden_{i+1}_{j}")
            else:
                for k in range(nn.neurons_hidden_layers[i-1]):
                    G.add_edge(f"hidden_{i}_{k}", f"hidden_{i+1}_{j}")
        if i+1 < len(nn.neurons_hidden_layers):
            y = nn.neurons_hidden_layers[i+1]/2
        else:
            y = nn.neurons_output/2
        x += 1

    # add output layer
    for i in range(nn.neurons_output):
        G.add_node(f"output_{i}", layer="output", neurons=nn.neurons_output)
        pos[f"output_{i}"] = (x, y)
        y -= 1
        for j in range(nn.neurons_hidden_layers[-1]):
            G.add_edge(f"hidden_{len(nn.neurons_hidden_layers)}_{j}", f"output_{i}")

    nx.draw(G, pos, with_labels=False)
    labels = nx.get_node_attributes(G, 'neurons')
    nx.draw_networkx_labels(G, pos, labels)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis((xmin, xmax, ymin, ymax))
    plt.show(block=True)  # adiciona esta linha para evitar a abertura de outra janela



class NeuralNetwork:
    def __init__(self, neurons_input, neurons_hidden_layers, neurons_output):
        self.neurons_hidden_layers = neurons_hidden_layers
        self.neurons_input = neurons_input
        self.neurons_output = neurons_output
        self.weights = []
        self.biases = []
        self.weights.append(np.random.rand(neurons_input, neurons_hidden_layers[0]))
        self.biases.append(np.random.rand(neurons_hidden_layers[0]))
        for i in range(1, len(neurons_hidden_layers)):
            self.weights.append(np.random.rand(neurons_hidden_layers[i-1], neurons_hidden_layers[i]))
            self.biases.append(np.random.rand(neurons_hidden_layers[i]))
        self.weights.append(np.random.rand(neurons_hidden_layers[-1], neurons_output))
        self.biases.append(np.random.rand(neurons_output))

    def feedforward(self, input):
        self.layers = [input]
        for i in range(len(self.neurons_hidden_layers)):
            self.layers.append(sigmoid(np.dot(self.layers[i], self.weights[i]) + self.biases[i]))
        self.output = sigmoid(np.dot(self.layers[-1], self.weights[-1]) + self.biases[-1])
        return self.output

    def backprop(self, input, y):
        # application of the chain rule to find derivative of the loss function with respect to weights and biases
        d_weights = [0] * (len(self.neurons_hidden_layers) + 1)
        d_biases = [0] * (len(self.neurons_hidden_layers) + 1)
        delta = (self.output - y) * sigmoid_derivative(self.output)
        d_weights[-1] = np.dot(self.layers[-1].T, delta)
        d_biases[-1] = np.mean(delta, axis=0)
        for i in range(len(self.neurons_hidden_layers), 0, -1):
            delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.layers[i])
            d_weights[i-1] = np.dot(self.layers[i-1].T, delta)
            d_biases[i-1] = np.mean(delta, axis=0)
        # update the weights and biases with the derivative (slope) of the loss function
        for i in range(len(self.neurons_hidden_layers) + 1):
            self.weights[i] -= d_weights[i]
            self.biases[i] -= d_biases[i]

    def train(self, x, y, num_iterations):
        for i in range(num_iterations):
            self.feedforward(x)
            self.backprop(x, y)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

# set the number of neurons in the input, hidden layers and output layers
neurons_input = 3
neurons_hidden_layers = [4,8,4]
neurons_output = 1
nn = NeuralNetwork(neurons_input, neurons_hidden_layers, neurons_output)

nn.train(X, y, 1500)

visualize_nn(nn)

while True:
    input_str = input("Enter input values separated by commas (or 'q' to quit): ")
    if input_str == 'q':
        break
    input_values = [float(x) for x in input_str.split(',')]
    input_values = np.array(input_values)
    print("Output: ", nn.feedforward(input_values))
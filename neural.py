import random
from engine import Value
from loss_fn import mse_loss, binary_cross_entropy
from optimizers import SGD

class neural_network:

    class Neuron:
        def __init__(self, input_dim, activation_fn):
            self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
            self.biases = Value(random.uniform(-1, 1))
            self.activation_fn = activation_fn
    
        def __call__(self, x):
            out = self.biases
            for i in range(len(x)):
                out += x[i] * self.weights[i]
            return self.activation_fn(out)

    class Layer:
        def __init__(self, input_dim, output_dim, activation_fn):
            self.neurons = [neural_network.Neuron(input_dim, activation_fn) for _ in range(output_dim)]
        
        def __call__(self, x):
            return [neuron(x) for neuron in self.neurons]

    def __init__(self, layers):
        self.layers = layers
        self.parameters = [p for layer in layers for neuron in layer.neurons for p in neuron.weights + [neuron.biases]]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, inputs, targets, epochs, loss_fn, optimizer):
        for epoch in range(epochs):
            for x, y in zip(inputs, targets):
                preds = self(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data}")


def relu(x):
    return x.relu()

def sigmoid(x):
    return x.sigmoid()

def tanh(x):
    return x.tanh()

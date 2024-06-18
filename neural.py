import random
from engine import Value

class neural_network:

    class Neuron:
        def __init__(self, input_dim, activation_fn):
            self.weights = [Value(random.uniform(-1, 1)) for i in range(input_dim)]
            self.biases = Value(random.uniform(-1, 1))
            self.activation_fn = activation_fn
    
        def __call__(self, x):
            out = self.biases
            for i in range(len(x)):
                out += x[i] * self.weights[i]
            return Value.activation_fn(out)

    class Layer:
        class Layer:
            def __init__(self, input_dim, output_dim, activation_fn):
                self.neurons = [neural_network.Neuron(input_dim, activation_fn) for _ in range(output_dim)]
        
            def __call__(self, x):
                return [neuron(x) for neuron in self.neurons]

    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

            
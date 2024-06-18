import math

class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = []

    def __add__(self, other):
        out = Value(self.data + other.data)
        
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data)
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other.data)
        
        def _backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            other.grad += (self.data ** other.data) * math.log(self.data) * out.grad
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def __sub__(self, other):
        out = Value(self.data - other.data)
        
        def _backward():
            self.grad += 1 * out.grad
            other.grad -= 1 * out.grad
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def relu(self):
        out = Value(0 if self.data <= 0 else self.data)
        
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        
        out._backward = _backward
        out._prev = [self]
        return out

    def exp(self):
        out = Value(math.exp(self.data))
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def sigmoid(self):
        exp_val = self.exp()
        out = Value(1 / (1 + exp_val.data))
        
        def _backward():
            sigmoid_grad = out.data * (1 - out.data)
            self.grad += sigmoid_grad * out.grad
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def tanh(self):
        exp_p = self.exp()
        exp_n = Value(-self.data).exp()
        out = (exp_p - exp_n) / (exp_p + exp_n)
        
        def _backward():
            tanh_grad = 1 - (out.data ** 2)
            self.grad += tanh_grad * out.grad
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1
        for v in reversed(topo):
            v._backward()

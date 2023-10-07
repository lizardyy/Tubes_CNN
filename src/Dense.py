import numpy as np
class Dense:
    def __init__(self, num_units, activation_function, input_size = None):
        self.output_size = (num_units,)

        self.input_size = input_size
        self.num_units = num_units
        self.activation_function = activation_function

        # Initialize weights and bias with random values
        if (input_size != None):
            self.weights = np.random.randn(self.input_size, self.num_units)
            self.gradients = np.zeros((self.input_size, self.num_units))

        self.bias = np.zeros(self.num_units)
        
        # Store last input, output, deltas, and gradients
        self.input = None
        self.net = None
        self.output = None
        self.deltas = np.zeros(self.num_units)
        self.bias_update = np.zeros(self.num_units)

    def forward(self, input_data):
        self.input = input_data

        if (self.input_size is None):
            self.input_size = input_data.shape[0]
            self.weights = np.random.randn(self.input_size, self.num_units)
            self.gradients = np.zeros((self.input_size, self.num_units))
        
        # Perform matrix multiplication (input_data * weights) and add bias
        pre_activation = np.dot(input_data, self.weights) + self.bias

        # Store the pre-activation (net)
        self.net = pre_activation

        # Apply activation function based on the chosen mode
        if self.activation_function == "relu":
            output_data = self.relu(pre_activation)
        elif self.activation_function == "sigmoid":
            output_data = self.sigmoid(pre_activation)
        else:
            output_data =  pre_activation

        self.output = output_data
        return output_data

    def backward(self, front_deltas=None, label=None, front_weights=None):
        if front_deltas is None and label is None:
            raise ValueError("if deltas is None (i.e. output layer), label must be provided")
        if front_deltas is not None and front_weights is None:
            raise ValueError("if deltas is not None (i.e. hidden layer), front_weights must be provided")
        
        if self.activation_function == "relu":
            derivatives = self.relu_derivative(self.net)
        elif self.activation_function == "sigmoid":
            derivatives = self.sigmoid_derivative(self.net)
        else:
            derivatives = np.copy(self.net)

        if front_deltas is None:
            errors = [label[i] - self.output[i] for i in range(self.num_units)]
            self.deltas = np.multiply(errors, derivatives)
        else:
            self.deltas = np.zeros(self.num_units)
            for i in range(self.num_units):
                sum_delta = 0
                for k in range(front_weights.shape[1]):
                    sum_delta += front_deltas[k] * front_weights[i, k]
                self.deltas[i] = derivatives[i] * sum_delta
        
        for k in range(self.num_units):
            self.gradients[:, k] += -np.multiply(self.deltas[k], self.input)
        self.bias_update += -self.deltas

        return self.deltas

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.gradients
        self.bias -= learning_rate * self.bias_update
        self.set_gradients(0.)
        self.bias_update = np.zeros(self.num_units)

    def set_deltas(self, delta=0.):
        self.deltas = np.full(self.num_units, delta)

    def set_gradients(self, gradient=0.):
        if self.input_size is not None:
            self.gradients = np.full((self.input_size, self.num_units), gradient)


    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu_derivative(self, x):
        return (x > 0) * 1.
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def getModel(self):
        weight_list = self.weights.tolist()
        bias_list = self.bias.tolist()
        model = {
            "type": "dense",
            "params":{
                "kernel": weight_list ,
                "bias": bias_list,
                "activation": self.activation_function
            }
        }

        return model
    
    def setModel(self, modelJson):
        self.weights = np.array(modelJson["params"]["kernel"])
        self.bias = np.array(modelJson["params"]["bias"])
    
    def setWeights(self, weights):
        self.input_size = weights.shape[0]
        self.weights = weights

    def setBias(self, bias):
        self.bias = bias
        
    def showModel(self, input =1):
        print(f"Dense                ({self.num_units})                 {(input + 1) * self.num_units}")
        print("________________________________________________________")
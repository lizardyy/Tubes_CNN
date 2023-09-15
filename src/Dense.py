import numpy as np
class Dense:
    def __init__(self, input_size, num_units, activation_function):
        self.input_size = input_size
        self.num_units = num_units
        self.activation_function = activation_function

        # Initialize weights and bias with random values
        self.weights = np.random.randn(input_size, num_units)
        self.bias = np.zeros(num_units)

    def forward(self, input_data):
        # Perform matrix multiplication (input_data * weights) and add bias
        pre_activation = np.dot(input_data, self.weights) + self.bias

        # Apply activation function based on the chosen mode
        if self.activation_function == "relu":
            output_data = self.relu(pre_activation)
        elif self.activation_function == "sigmoid":
            output_data = self.sigmoid(pre_activation)
        else:
            raise ValueError("Invalid activation function. Choose 'relu' or 'sigmoid'.")

        return output_data

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
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
    
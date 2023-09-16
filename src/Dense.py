import numpy as np
class Dense:
    def __init__(self, num_units, activation_function, input_size = None):
        self.input_size = input_size
        self.num_units = num_units
        self.activation_function = activation_function

        # Initialize weights and bias with random values
        if (input_size != None):
            self.weights = np.random.randn(self.input_size, self.num_units)

        self.bias = np.zeros(num_units)

    def forward(self, input_data):

        if (self.input_size == None):
            self.input_size = input_data.shape[0]
            self.weights = np.random.randn(self.input_size, self.num_units)
        
        # Perform matrix multiplication (input_data * weights) and add bias
        pre_activation = np.dot(input_data, self.weights) + self.bias

        # Apply activation function based on the chosen mode
        if self.activation_function == "relu":
            output_data = self.relu(pre_activation)
        elif self.activation_function == "sigmoid":
            output_data = self.sigmoid(pre_activation)
        else:
            output_data =  pre_activation
    
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
    
    def setWeights(self, weights):
        self.input_size = weights.shape[0]
        self.weights = weights

    def setBias(self, bias):
        self.bias = bias
        
    def showModel(self, input =1):
        print(f"Dense                ({self.num_units})                 {(input + 1) * self.num_units}")
        print("________________________________________________________")
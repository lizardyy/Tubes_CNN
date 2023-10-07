import numpy as np
class Flatten:
    def __init__(self):
        self.output_shape = None

        self.output = None
        self.input = None

    def forward(self, input_data):
        self.input = input_data

        # Mendapatkan ukuran input
        input_shape = input_data.shape

        # Melakukan flatten, mengubah tensor tiga dimensi menjadi tensor dua dimensi
        output_data = input_data.flatten()
        self.output_shape = output_data.shape
        self.output = output_data
        return output_data

    def backward(self, front_deltas=None, label=None, front_weights=None):
        self.deltas = np.zeros(self.output.shape[0])
        for i in range(self.output.shape[0]):
            sum_delta = 0
            for k in range(front_weights.shape[1]):
                sum_delta += front_deltas[k] * front_weights[i, k]
            self.deltas[i] = sum_delta

        self.weights = np.reshape(front_weights, (front_weights.shape[1], self.input.shape[0], self.input.shape[1], self.input.shape[2]))
        self.deltas = np.reshape(self.deltas, self.input.shape)
        return self.deltas
    
    def getModel(self):
        model = {
            "type": "flatten",
            "params": {}
        }
        return model
    
    def showModel(self, input =1):
        print(f"Flatten              {self.output_shape}                  0 ")
        print("________________________________________________________")
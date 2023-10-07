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
        self.weights = np.reshape(front_weights, self.input.shape)
        self.deltas = np.reshape(front_deltas, self.input_shape)
    
    def getModel(self):
        model = {
            "type": "flatten",
            "params": {}
        }
        return model
    
    def showModel(self, input =1):
        print(f"Flatten              {self.output_shape}                  0 ")
        print("________________________________________________________")
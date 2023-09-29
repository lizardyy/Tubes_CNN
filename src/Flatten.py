import numpy as np
class Flatten:
    def __init__(self):
        self.output_shape = None

        self.output = None
        self.input = None

    def forward(self, input_data):
        # Mendapatkan ukuran input
        input_shape = input_data.shape

        # Melakukan flatten, mengubah tensor tiga dimensi menjadi tensor dua dimensi
        output_data = input_data.flatten()
        self.output_shape = output_data.shape
        self.output = output_data
        return output_data
    
    def getModel(self):
        model = {
            "type": "flatten",
            "params": {}
        }
        return model
    
    def showModel(self, input =1):
        print(f"Flatten              {self.output_shape}                  0 ")
        print("________________________________________________________")
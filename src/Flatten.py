import numpy as np
class Flatten:
    def __init__(self):
        pass

    def forward(self, input_data):
        # Mendapatkan ukuran input
        input_shape = input_data.shape

        # Melakukan flatten, mengubah tensor tiga dimensi menjadi tensor dua dimensi
        output_data = input_data.flatten()

        return output_data
    
    def getModel(self):
        return ''

    
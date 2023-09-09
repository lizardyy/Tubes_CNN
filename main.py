from src import Convolution
import numpy as np
input_size = (32, 32, 3)
input_data = np.random.randn(*input_size)




conv_layer = Convolution(input_size=(256, 256, 3), padding_size=1, filter_size=(3, 3), num_filters=1, stride=1, biases=0)

output = conv_layer.forward(input_data )
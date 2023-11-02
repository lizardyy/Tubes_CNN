import json
import numpy as np
import math

from src.Convolution import Convolution
from src.Pooling import Pooling
from src.Dense import Dense
from src.Flatten import Flatten
from src.LSTM import LSTM

class Model: 
    def __init__(self):
        self.layers =[]

    def predict(self, X):
        results = []
        for x in X:
            input = x
            output = None
            for layer in self.layers:
                output = layer.forward(input)
                input = output
            results.append(output)
        return np.array(results)
    
    def fit(self, X=None, y=None, epochs=1, batch_size=32, learning_rate=0.001):
        if X is None or y is None:
            raise ValueError("x and y must be provided and not None")
        if len(X) != len(y):
            raise ValueError("length of x and y must be equal")
        
        data_count = len(X)
        for epoch in range(epochs):
            batchs_total = math.ceil( data_count / batch_size )
            for batch in range(batchs_total):
                batch_x = X[batch * batch_size : (batch + 1) * batch_size]
                batch_y = y[batch * batch_size : (batch + 1) * batch_size]
                batch_length = len(batch_x)

                for i in range(batch_length):
                    label = batch_y[i]
                    if len(y.shape) == 1:   # In the label is not a list, convert it to list
                        label = np.array([label])
                    input = batch_x[i]

                    # Forward
                    output = None
                    for layer in self.layers:
                        output = layer.forward(input)
                        input = output
                    
                    # Backward: calculate delta and gradient
                    front_deltas = None
                    front_weights = None
                    for l in range(len(self.layers)-1, -1, -1):
                        layer = self.layers[l]
                        front_deltas = layer.backward(front_deltas=front_deltas, label=label, front_weights=front_weights)
                        if isinstance(layer, Convolution):
                            front_weights = layer.filter
                        else:
                            front_weights = layer.weights
                
                # Update weights
                for layer in self.layers:
                    layer.update_weights(learning_rate)
            
            # End of an epoch
            print(f"===== Epoch {epoch+1} =====")

            # Calculate accuracy (only for dense sigmoid with one output unit)
            if isinstance(self.layers[-1], Dense) and self.layers[-1].num_units == 1:
                N = len(X)
                yp = self.predict(X)
                temp = np.zeros((y.shape[0], 1))
                if len(y.shape) == 1:
                    for i in range(len(y)):
                        temp[i][0] = y[i]
                y = temp
                
                correct = 0
                for i in range(N):
                    if len(y.shape) == 1:
                        if round(y[i]) == round(yp[i]):
                            correct += 1
                    else:
                        if round(y[i][0]) == round(yp[i][0]):
                            correct += 1
                accuracy = correct / N
                print(f"Accuracy: {accuracy}")
                        
        
    def add(self, layer):
        if (len(self.layers) > 0):
            layer.input_shape = self.layers[-1].output_shape
        self.layers.append(layer)

    def saveModel(self):
        save_model = []
        for layer in self.layers:
            out_model = layer.getModel()
            if out_model != '':
                save_model.append(out_model)
        file_name = './Model/model.json'

        # Open the file in write mode and write the save_model list to it as JSON
        with open(file_name, 'w') as json_file:
            json.dump(save_model, json_file)
    
    # To be continued on Milestone B
    @staticmethod
    def loadModel(file_name):
        model = Model()
        filename = './Model/' +file_name
        with open(filename, 'r') as json_file:
            layers = json.load(json_file)
            for layer in layers:
                l = None
                match layer["type"]:
                    case "conv2d":
                        num_kernel = len(layer["params"]["kernel"])
                        kernel = layer["params"]["kernel"][0]
                        depth = len(kernel)
                        rows = len(kernel[0])
                        cols = len(kernel[0][0])
                        l = Convolution((256,256,3), 0, (rows,cols,depth), num_kernel, 1)
                        l.setModel(layer)
                    case "dense":
                        num_units = len(layer["params"]["kernel"])
                        l = Dense(num_units, "relu")
                        l.setModel(layer)
                    case "max_pooling2d":
                        l = Pooling((2,2), 1, "max")
                    case "average_pooling2d":
                        l = Pooling((2,2), 1, "average")
                    case "flatten":
                        l = Flatten()
                    case "lstm":
                        l = LSTM((2,2), 1)
                        l.setModel(layer)

                model.add(l)
        return model

    def summary(self):
        lwidth = 20
        owidth = 20
        pwidth = 15

        print('Model: "sequential"')
        print("_" * (lwidth + owidth + pwidth))

        print(f"{'Layer (type)':<{lwidth}}{'Output Shape':<{owidth}}{'Param #':<{pwidth}}")
        print("=" * (lwidth + owidth + pwidth))

        total_params = 0
        for layer in self.layers:
            total_params += layer.summary(lwidth, owidth, pwidth)
        print("=" * (lwidth + owidth + pwidth))

        print(f"Total params: {total_params}")
        print("=" * (lwidth + owidth + pwidth))

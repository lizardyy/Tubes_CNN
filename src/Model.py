import json
import numpy as np


class Model: 
    def __init__(self):
        self.layers =[]

    def train_network(self,train_data, label_data, batch_size, lr=0.01, epochs=200):
        batch_train = np.array_split(train_data,batch_size)
        batch_label = np.array_split(label_data, batch_size)
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            num = 0

            for i in range(len(batch_train)):
                # for j in range(len(batch_train[i])):
                for j in range(5):
                    input = batch_train[i][j]
                    for layer in self.layers:
                        output = layer.forward(input)
                        input = output
                    predictions = output[0]
                    if (predictions>0.5):
                        predict = 1
                    else :
                        predict = 0
                    num +=1
                    if predict == batch_label[i][j]:
                        correct_predictions += 1
            accuracy = (correct_predictions / num) * 100.0
            print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.2f}%")    
                
        return 0
        
    def add(self,layer):
        self.layers.append(layer)

    def saveModel(self):
        save_model = []
        for layer in self.layers:
            out_model = layer.getModel()
            if out_model != '':
                save_model.append(out_model)
        file_name = 'model.json'

        # Open the file in write mode and write the save_model list to it as JSON
        with open(file_name, 'w') as json_file:
            json.dump(save_model, json_file)
    
    # To be continued on Milestone B
    # @staticmethod
    # def loadModel(file_name):
    #     model = Model()
    #     with open(file_name, 'r') as json_file:
    #         layers = json.load(json_file)
    #         for layer in layers:
    #             l = None
    #             match layer["type"]:
    #                 case "conv2d":
    #                     l = Convolution((256,256,3), 0, (3,3), 1, 1, None)
    #                     l.setModel(layer)
    #                 case "dense":
    #                     l = Dense(1, "relu")
    #                     l.setModel(layer)
    #                 case "max_pooling2d":
    #                     l = Pooling((2,2), 1, "max")
    #                 case "average_pooling2d":
    #                     l = Pooling((2,2), 1, "average")
    #                 case "flatten":
    #                     l = Flatten()
    #             model.add(l)
    #     return model

    def showModel(self):
        print("Layer (type)         Output Shape             Param #")
        print("========================================================")
        for layer in self.layers:
            layer.showModel()
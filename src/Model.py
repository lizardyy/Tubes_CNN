class Model: 
    def __init__(self):
        self.layers =[]

    def train_network(self,train_data, lr=0.01, epochs=200):
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            num = 0

            # for i in range(1):
            # for i in range(len(train_data)):
            for i, (batch_x, batch_y) in enumerate(train_data):
                # Forward pass
                for j in range(len(batch_x)):
                    for i in range(len(self.layers)):
                        self.layers[i].forward(batch_x[j], batch_y[])
                    
                        predictions = dense_sigmoid_output
                        if (predictions>0.5):
                            predict = 1
                        else :
                            predict = 0
                        # Hitung prediksi
                        if predict == batch_y[j]:
                            correct_predictions += 1
                        num +=1
            accuracy = (correct_predictions / num) * 100.0
            print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.2f}%")
        return dense_sigmoid_output 
        
    def add(self,layer):
        self.Layers.add(layer)

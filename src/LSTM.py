import numpy as np

class LSTM:
    def __init__(self, input_shape, num_units, activation=None):
        self.input_shape = input_shape
        self.output_shape = (None, num_units)

        _, num_feature = input_shape
        self.num_feature = num_feature
        self.num_units = num_units

        # Weight matrices and biases for forget gate
        self.U_f = np.random.rand(num_feature, num_units)
        self.W_f = np.random.rand(num_units, num_units)
        self.b_f = np.zeros((1, num_units))

        # Weight matrices and biases for input gate
        self.U_i = np.random.rand(num_feature, num_units)
        self.W_i = np.random.rand(num_units, num_units)
        self.b_i = np.zeros((1, num_units))

        # Weight matrices and biases for cell state
        self.U_c = np.random.rand(num_feature, num_units)
        self.W_c = np.random.rand(num_units, num_units)
        self.b_c = np.zeros((1, num_units))

        # Weight matrices and biases for output gate
        self.U_o = np.random.rand(num_feature, num_units)
        self.W_o = np.random.rand(num_units, num_units)
        self.b_o = np.zeros((1, num_units))

        # Activation function, default 'relu'
        self.activation = 'relu'
        if activation != None:
            self.activation = activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        # Initial cell state and hidden state
        self.cell_state = np.zeros((1, self.num_units))
        self.hidden_state = np.zeros((1, self.num_units))
        for x in X:
            # forget gate
            forget_gate = self.sigmoid(np.dot(x, self.U_f) + np.dot(self.hidden_state, self.W_f) + self.b_f)

            # input gate
            input_gate = self.sigmoid(np.dot(x, self.U_i) + np.dot(self.hidden_state, self.W_i) + self.b_i)

            # candidate cell state
            candidate_cell_state = self.tanh(np.dot(x, self.U_c) + np.dot(self.hidden_state, self.W_c) + self.b_c)

            # cell state
            self.cell_state = forget_gate * self.cell_state + input_gate * candidate_cell_state

            # output gate
            output_gate = self.sigmoid(np.dot(x, self.U_o) + np.dot(self.hidden_state, self.W_o) + self.b_o)
            
            # hidden state
            net = output_gate * self.tanh(self.cell_state)
            if self.activation == 'relu':
                self.hidden_state = self.relu(net)
            elif self.activation == 'sigmoid':
                self.hidden_state = self.sigmoid(net)
            else:
                self.hidden_state = net
        return self.hidden_state.flatten()
    
    def set_weight_hidden(self,state, weight):
        if (state == 'forget'):
            self.W_f = weight
        elif(state=='input'):
            self.W_i = weight
        elif(state=='cell'):
            self.W_c = weight
        elif (state == 'output'):
            self.W_o = weight
        else:
            print("Invalid state")
    
    def set_weight_input(self,state, weight):
        if (state == 'forget'):
            self.U_f = weight
        elif(state=='input'):
            self.U_i = weight
        elif(state=='cell'):
            self.U_c = weight
        elif (state == 'output'):
            self.U_o = weight
        else:
            print("Invalid state")

    def set_bias(self,state, bias):
        if (state == 'forget'):
            self.b_f = bias
        elif(state=='input'):
            self.b_i = bias
        elif(state=='cell'):
            self.b_c = bias
        elif (state == 'output'):
            self.b_o = bias
        else:
            print("Invalid state")

    def getModel(self):
        model = {
            "type": "lstm",
            "params":{
                "U_i": self.U_i.tolist(),
                "U_f": self.U_f.tolist(),
                "U_c": self.U_c.tolist(),
                "U_o": self.U_o.tolist(),

                "W_i": self.W_i.tolist(),
                "W_f": self.W_f.tolist(),
                "W_c": self.W_c.tolist(),
                "W_o": self.W_o.tolist(),

                "b_i": self.b_i.reshape(-1).tolist(),
                "b_f": self.b_f.reshape(-1).tolist(),
                "b_c": self.b_c.reshape(-1).tolist(),
                "b_o": self.b_o.reshape(-1).tolist(),

                "activation": self.activation
            }
        }

        return model

    # Pada load model berikut pada contoh spek yang diberikan U merupakan weight untuk hidden sedangkan W merupakan weight untuk input
    # Namun pada model yang kami buat adalah sebaliknya (berpatokan pada power point) dimana W untuk hidden dan U untuk input
    # Untuk itu kami merubah json contoh
    def setModel(self, modelJson):
        U_i = np.array(modelJson['params']["U_i"])
        self.set_weight_input("input",U_i)
        U_f = np.array(modelJson['params']["U_f"])
        self.set_weight_input("forget",U_f)
        U_c = np.array(modelJson['params']["U_c"])
        self.set_weight_input("cell",U_c)
        U_o = np.array(modelJson['params']["U_o"])
        self.set_weight_input("output",U_o)

        W_i = np.array(modelJson['params']["W_i"])
        self.set_weight_hidden("input",W_i)
        W_f = np.array(modelJson['params']["W_f"])
        self.set_weight_hidden("forget",W_f)
        W_c = np.array(modelJson['params']["W_c"])
        self.set_weight_hidden("cell",W_c)
        W_o = np.array(modelJson['params']["W_o"])
        self.set_weight_hidden("output",W_o)

        b_i = np.array(modelJson['params']["b_i"])
        self.set_bias("input", b_i.reshape(1, -1))
        b_f = np.array(modelJson['params']["b_f"])
        self.set_bias("forget", b_f.reshape(1, -1))
        b_c = np.array(modelJson['params']["b_c"])
        self.set_bias("cell", b_c.reshape(1, -1))
        b_o = np.array(modelJson['params']["b_o"])
        self.set_bias("output", b_o.reshape(1, -1))

        self.num_feature = U_f.shape[0]
        self.num_units = b_o.shape[0]
        self.output_shape = (None, self.num_units)
        if 'activation' in modelJson['params']:
            self.activation = modelJson['params']['activation']

        # Initial cell state and hidden state
        self.cell_state = np.zeros((1, self.num_units))
        self.hidden_state = np.zeros((1, self.num_units))

    def summary(self, lwidth, owidth, pwidth):
        num_params =  4 * self.num_units * (self.num_units + self.num_feature + 1)
        print(f"{'lstm (LSTM)':<{lwidth}}{f'{self.output_shape}':<{owidth}}{num_params:<{pwidth}}")
        return num_params

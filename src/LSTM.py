import numpy as np

class LSTM:
    def __init__(self, input_size, num_units):
        self.input_size = input_size
        self.num_units = num_units
        
        # Weight matrices and biases for forget gate
        self.U_f = np.random.rand(input_size, num_units)
        self.W_f = np.random.rand(num_units, num_units)
        self.b_f = np.zeros((1, num_units))
        
        # Weight matrices and biases for input gate
        self.U_i = np.random.rand(input_size, num_units)
        self.W_i = np.random.rand(num_units, num_units)
        self.b_i = np.zeros((1, num_units))
        
        # Weight matrices and biases for cell state
        self.U_c = np.random.rand(input_size, num_units)
        self.W_c = np.random.rand(num_units, num_units)
        self.b_c = np.zeros((1, num_units))
        
        # Weight matrices and biases for output gate
        self.U_o = np.random.rand(input_size, num_units)
        self.W_o = np.random.rand(num_units, num_units)
        self.b_o = np.zeros((1, num_units))
        
        # Initial cell state and hidden state
        self.cell_state = np.zeros((1, num_units))
        self.hidden_state = np.zeros((1, num_units))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        x = x.reshape(1, -1) 

        # forget gate
        forget_gate = self.sigmoid(np.dot(x, self.U_f) + np.dot(self.hidden_state, self.W_f) + self.b_f)
        print(forget_gate)

        # input gate
        input_gate = self.sigmoid(np.dot(x, self.U_i) + np.dot(self.hidden_state, self.W_i) + self.b_i)
        print(input_gate)
        # candidate cell state
        candidate_cell_state = self.tanh(np.dot(x, self.U_c) + np.dot(self.hidden_state, self.W_c) + self.b_c)
        print(candidate_cell_state)

        # cell state
        self.cell_state = forget_gate * self.cell_state + input_gate * candidate_cell_state
        
        # output gate
        output_gate = self.sigmoid(np.dot(x, self.U_o) + np.dot(self.hidden_state, self.W_o) + self.b_o)
        print(output_gate)
        
        # hidden state
        self.hidden_state = output_gate * self.tanh(self.cell_state)
        
        return self.cell_state, self.hidden_state
    
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
import numpy as np

# weight_{state} merupakan gabungan U (weight associated with the input) dan W (weight associated with the hidden state)
class LSTM:
    def __init__(self, input_size, num_units):
        self.input_size = input_size
        self.num_units = num_units
        
        # Weight matrices and biases for forget gate
        self.weights_forget = np.random.rand(input_size + num_units, num_units)
        self.bias_forget = np.zeros((1, num_units))
        
        # Weight matrices and biases for input gate
        self.weights_input = np.random.rand(input_size + num_units, num_units)
        self.bias_input = np.zeros((1, num_units))
        
        # Weight matrices and biases for cell state
        self.weights_cell = np.random.rand(input_size + num_units, num_units)
        self.bias_cell = np.zeros((1, num_units))
        
        # Weight matrices and biases for output gate
        self.weights_output = np.random.rand(input_size + num_units, num_units)
        self.bias_output = np.zeros((1, num_units))
        
        # Initial cell state and hidden state
        self.cell_state = np.zeros((1, num_units))
        self.hidden_state = np.zeros((1, num_units))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        x = x.reshape(1, -1) 
        concat_input = np.hstack((x, self.hidden_state))
        # forget gate
        forget_gate = self.sigmoid(np.dot(concat_input, self.weights_forget) + self.bias_forget)
        print(forget_gate)
        # input gate
        input_gate = self.sigmoid(np.dot(concat_input, self.weights_input) + self.bias_input)
        print(input_gate)
        # candidate cell state
        candidate_cell_state = self.tanh(np.dot(concat_input, self.weights_cell) + self.bias_cell)
        print(candidate_cell_state)
        # cell state
        self.cell_state = forget_gate * self.cell_state + input_gate * candidate_cell_state
        
        # output gate
        output_gate = self.sigmoid(np.dot(concat_input, self.weights_output) + self.bias_output)
        print(output_gate)
        
        # hidden state
        self.hidden_state = output_gate * self.tanh(self.cell_state)
        
        return self.cell_state, self.hidden_state
    
    def set_weight(self,state, weight):
        if (state == 'forget'):
            self.weights_forget = weight
        elif(state=='input'):
            self.weights_input = weight
        elif(state=='cell'):
            self.weights_cell = weight
        elif (state == 'output'):
            self.weights_output = weight
        else:
            print("Invalid state")
    
    def set_bias(self,state, bias):
        if (state == 'forget'):
            self.bias_forget = bias
        elif(state=='input'):
            self.bias_input = bias
        elif(state=='cell'):
            self.bias_cell = bias
        elif (state == 'output'):
            self.bias_output = bias
        else:
            print("Invalid state")
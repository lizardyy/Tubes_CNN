import numpy as np

class Convolution:
    
    # TODO: diferent width / height
    def __init__(self, input_size, padding_size, filter_size, num_filters, stride):
        self.input_size = input_size
        self.padding_size = int(padding_size)
        self.filter_size = filter_size

        self.num_filters = num_filters
        self.stride = int(stride)

        self.output_size = (((input_size[0] - filter_size[0] + 2 * padding_size) // stride ) + 1, ((input_size[1] - filter_size[1] + 2 * padding_size) // stride ) + 1)
        # init random filter
        self.filter = [np.random.randn(self.filter_size[0], self.filter_size[1], input_size[2]) for _ in range(self.num_filters)]
        self.bias = np.zeros(self.num_filters)

        # Store last input, output, deltas, and gradients
        self.input = None
        self.net = None
        self.output = None
        self.deltas = np.zeros((self.output_size[0], self.output_size[1], self.num_filters))
        self.bias_update = np.zeros(self.num_filters)
        self.gradients = np.zeros((self.num_filters, self.filter_size[0], self.filter_size[1], self.filter_size[2]))

    # include convolution and detector
    def forward(self, input):
        self.input = input

        if self.input_size is not None and input.shape != self.input_size:
            raise TypeError(f"input size doesn't match! must be {self.input_size}")


        # Menambahkan padding jika diperlukan
        if self.padding_size > 0:
            padded_input = np.pad(input, ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size), (0, 0)), mode='constant')
        else:
            padded_input = input

        # init input
        self.net = np.zeros((self.output_size[0], self.output_size[1], self.num_filters))
        output = np.zeros((self.output_size[0], self.output_size[1], self.num_filters))

        for i in range(0,self.input_size[0] - self.filter_size[0] + (2 * self.padding_size) + 1, self.stride):
            for j in range(0, self.input_size[1] - self.filter_size[1] + (2 * self.padding_size) + 1,self.stride):
                for n in range(self.num_filters):
                    # Mengambil bagian input yang sesuai dengan ukuran filter
                    input_patch = padded_input[i:i+self.filter_size[0], j:j+self.filter_size[1]]

                    # Melakukan operasi konvolusi
                    self.net[i//self.stride][j//self.stride][n] = np.sum(input_patch * self.filter[n]) + self.bias[n]
                    
                    # Melakukan fungsi aktivasi (relu)
                    output[i//self.stride][j//self.stride][n] = np.maximum(0, self.net[i//self.stride][j//self.stride][n])

        self.output = output
        return output
    
    def backward(self, front_deltas=None, label=None, front_weights=None):
        derivatives = self.relu_derivative(self.net)

        if front_deltas is None:
            errors = label - self.output
            self.deltas = np.multiply(errors, derivatives)

            if self.stride > 1:
                Md = (self.input_size[0] - self.filter_size[0] + 2*self.padding_size) + 1
                Nd = (self.input_size[1] - self.filter_size[1] + 2*self.padding_size) + 1
                pad_size = ((self.stride, self.stride))
                temp = np.zeros((Md, Nd, self.num_filters))
                for k in range(self.num_filters):
                    dilated = self.dilate(self.deltas[:,:,k], pad_size)
                    temp[:, :, k] += dilated[0:Md, 0:Nd]
                self.deltas = temp
        else:
            if front_weights is None:
                self.deltas = np.zeros((self.output_size[0], self.output_size[1], self.num_filters))
            elif self.stride == 1:
                self.deltas = np.zeros((self.output_size[0], self.output_size[1], self.num_filters))
                for k in range(self.num_filters):
                    for l in range(len(front_weights)):
                        self.deltas[:, :, k] += self.fullconv(front_deltas[:, :, l], front_weights[l, :, :, k])
                self.deltas = np.multiply(self.deltas, derivatives)
            else:
                Md = (self.input_size[0] - self.filter_size[0] + 2*self.padding_size) + 1
                Nd = (self.input_size[1] - self.filter_size[1] + 2*self.padding_size) + 1
                self.deltas = np.zeros((Md, Nd, self.num_filters))
                
                for k in range(self.num_filters):
                    for l in range(len(front_weights)):
                        conv = self.fullconv(front_deltas[:, :, l], front_weights[l, :, :, k])
                        pad_size = ((self.stride, self.stride))
                        convdilated = self.dilate(conv, pad_size)
                        self.deltas[:, :, k] += convdilated[0:Md, 0:Nd]
                self.deltas = np.multiply(self.deltas, derivatives)

        # Menambahkan padding jika diperlukan
        if self.padding_size > 0:
            padded_input = np.pad(self.input, ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size), (0, 0)), mode='constant')
        else:
            padded_input = self.input

        for i in range(self.num_filters):
            self.bias_update[i] += -np.sum(self.deltas[:, :, i])
            
        for k in range(self.filter_size[2]):
            for i in range(self.num_filters):
                self.gradients[i, :, :, k] += self.validconv(padded_input[:, :, k], self.rotate180(self.deltas[:, :, i]))
        
        return self.deltas
    
    def update_weights(self, learning_rate):
        self.filter -= learning_rate * self.gradients
        self.bias -= learning_rate * self.bias_update
        self.set_gradients(0.)
        self.bias = np.zeros(self.num_filters)

        
    def validconv(self, m1, m2):
        # Convolution: m2 is rotated 180 degree first
        m2 = self.rotate180(m2)

        m1_shape = m1.shape
        M = m1_shape[0]
        N = m1_shape[1]
        d = 1
        if (len(m1_shape) == 3):
            d = m1_shape[2]

        m2_shape = m2.shape
        m = m2_shape[0]
        n = m2_shape[1]

        if d == 1:
            result = np.zeros((M - m + 1, N - n + 1))
        else:
            result = np.zeros((M - m + 1, N - n + 1, d))

        for i in range(M - m + 1):
            for j in range(N - n + 1):
                m1_slc = m1[i : i+m, j : j+n]
                result[i][j] = np.multiply(m1_slc, m2).sum(axis=1).sum(axis=0)
        return result

    
    # Calculate full convolution
    def fullconv(self, m1, m2):
        # Convolution: m2 is rotated 180 degree first
        m2 = self.rotate180(m2)

        m1_shape = m1.shape
        M = m1_shape[0]
        N = m1_shape[1]
        d = 1
        if (len(m1_shape) == 3):
            d = m1_shape[2]

        m2_shape = m2.shape
        m = m2_shape[0]
        n = m2_shape[1]

        if d == 1:
            result = np.zeros((M + m - 1, N + n - 1))
        else:
            result = np.zeros((M + m - 1, N + n - 1, d))

        for i in range(M + m - 1):
            for j in range(N + n - 1):
                m1_slc = m1[max(i-m+1, 0) : i+1, max(j-n+1, 0) : j+1]
                m2_slc = m2[max(m-i-1, 0) : (M-i+m-1), max(n-j-1, 0) : (N-j+n-1)]
                result[i][j] = np.multiply(m1_slc, m2_slc).sum(axis=1).sum(axis=0)
        return result

    def rotate180(self, m):
        return np.rot90(m, 2)

    def relu(self, x):
        return (x > 0) * x

    def relu_derivative(self, x):
        return (x > 0) * 1.
    
    def set_deltas(self, delta=0.):
        self.deltas = np.full((self.output_size[0], self.output_size[1], self.num_filters), delta)

    def set_gradients(self, gradient=0.):
        self.gradients = np.full((self.num_filters, self.filter_size[0], self.filter_size[1], self.filter_size[2]), gradient)

    def dilate(self, x, dim):
        result_shape = np.multiply(x.shape, dim)
        result = np.zeros(result_shape, dtype=x.dtype)
        result[0:result_shape[0]:dim[0], 0:result_shape[1]:dim[1]] = x
        return result

    def getModel(self):
        filter_list = [filter_.tolist() for filter_ in self.filter]
        bias_list = self.bias.tolist()
        model = {
            "type": "conv2d",
            "params":{
                "kernel": filter_list,
                "bias": bias_list
            }
        }

        return model

    def setModel(self, modelJson):
        self.filter = [np.array(filter_) for filter_ in modelJson["params"]["kernel"]]
        self.bias = np.array(modelJson["params"]["bias"])

        # self.num_filters = len(modelJson["params"]["kernel"][0][0][0])
        # self.filter_size = len(modelJson["params"]["kernel"]), len(modelJson["params"]["kernel"][0])
        
    
    def setFilter(self, filter):
        self.filter = filter

    def showModel(self, input_size):
        output_shape = (self.output_size[0],self.output_size[1],self.num_filters)
        print(self.filter_size)
        print(f"Convolution          ({output_shape})            {(self.num_filters * (self.filter_size[0] * self.filter_size[1] * self.filter_size[2] + 1))}")
        print("________________________________________________________")
        return output_shape
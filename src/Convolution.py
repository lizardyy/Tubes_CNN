import numpy as np

class Convolution:
    
    # TODO: diferent width / height
    def __init__(self, input_size, padding_size, filter_size, num_filters, stride, bias):
        self.input_size = input_size
        self.padding_size = padding_size
        self.filter_size = filter_size

        self.num_filters = num_filters
        self.stride = stride
        # self.bias = bias

        self.output_size = (((input_size[0] - filter_size[0] + 2 * padding_size) // stride ) + 1, ((input_size[1] - filter_size[1] + 2 * padding_size) // stride ) + 1)
        # init random filter
        self.filter = [np.random.randn(self.filter_size[0], self.filter_size[1], input_size[2]) for _ in range(self.num_filters)]
        self.bias = np.zeros(self.num_filters)

    # include convolution and detector
    def forward(self, input):
        if self.input_size != None and input.shape != self.input_size:
            raise TypeError(f"input size doesn't match! must be {self.input_size}")


        # Menambahkan padding jika diperlukan
        if self.padding_size > 0:
            padded_input = np.pad(input, ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size), (0, 0)), mode='constant')
        else:
            padded_input = input

        # init input
        output = np.zeros((self.output_size[0], self.output_size[1], self.num_filters))

        for i in range(0,self.input_size[0] - self.filter_size[0] + (2 * self.padding_size) + 1, self.stride):
            for j in range(0, self.input_size[1] - self.filter_size[1] + (2 * self.padding_size) + 1,self.stride):
                for n in range(self.num_filters):
                    # Mengambil bagian input yang sesuai dengan ukuran filter
                    input_patch = padded_input[i:i+self.filter_size[0], j:j+self.filter_size[1]]
                    # Melakukan operasi konvolusi
                    output[i//self.stride][j//self.stride][n] = np.maximum(0, np.sum(input_patch * self.filter[n]) + self.bias[n])

        return output
    
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

        self.num_filters = len(modelJson["params"]["kernel"][0][0][0])
        self.filter_size = len(modelJson["params"]["kernel"]), len(modelJson["params"]["kernel"][0])
        
    
    def setFilter(self, filter):
        self.filter = filter

    def showModel(self):
        print(f"Convolution          {self.output_size}             {(self.num_filters * (self.filter_size[0] * self.filter_size[0] + 1))}")
        print("________________________________________________________")
import numpy as np

class Convolution:
    
    # TODO: diferent width / height
    def __init__(self, input_size, padding_size, filter_size, num_filters, stride, biases):
        self.input_size = input_size
        self.padding_size = padding_size
        self.filter_size = filter_size

        self.num_filters = num_filters
        self.stride = stride
        self.biases = biases

        self.output_size = (((input_size[0] - filter_size[0] + 2 * padding_size) // stride ) + 1, ((input_size[1] - filter_size[1] + 2 * padding_size) // stride ) + 1)


    def convolution(self, input):

        # init random filter
        self.filter = [np.random.randn(self.filter_size[0], self.filter_size[1], input.shape[2]) for _ in range(self.num_filters)]

        # Menambahkan padding jika diperlukan
        if self.padding_size > 0:
            padded_input = np.pad(input, ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size), (0, 0)), mode='constant')
        else:
            padded_input = input

        # init input
        self.output = np.zeros((self.output_size[0], self.output_size[1], self.num_filters))

        for i in range(0,self.input_size[0] - self.filter_size[0] + (2 * self.padding_size) + 1, self.stride):
            for j in range(0, self.input_size[1] - self.filter_size[1] + (2 * self.padding_size) + 1,self.stride):
                for n in range(self.num_filters):
                    # Mengambil bagian input yang sesuai dengan ukuran filter
                    input_patch = padded_input[i:i+self.filter_size[0], j:j+self.filter_size[1]]
                    # Melakukan operasi konvolusi
                    self.output[i//self.stride][j//self.stride][n] = np.sum(input_patch * self.filter[n])

        return self.output

    def detector(self, input):
        self.output = np.maximum(0, input)
        return self.output
    
    def polling(self, input, filter_size, stride, mode):
        input_size = input.shape
        output_size = (((input_size[0] - filter_size[0] ) // stride ) + 1, ((input_size[1] - filter_size[1]) // stride ) + 1)
        output = np.zeros((output_size[0], output_size[1],input_size[2]))
        for i in range(0,input_size[0] - filter_size[0] +1, stride):
            for j in range(0, input_size[1] - filter_size[1] +1 ,stride):
                for n in range(input_size[2]):
                    input_patch = input[i:i+filter_size[0], j:j+filter_size[1], n]
                    if mode == "max":
                        # Max pooling
                        output[i//stride][j//stride][n] = np.max(input_patch)
                    elif mode == "average":
                        # Average pooling
                        output[i//stride][j//stride][n] = np.mean(input_patch)
            return output
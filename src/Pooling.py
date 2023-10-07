import numpy as np

class Pooling:
    def __init__(self,  filter_size, stride, mode):
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode
        self.output_size = None

    def forward(self, input):
        self.input = input

        input_size = input.shape
        output_size = (((input_size[0] - self.filter_size[0] ) // self.stride ) + 1, ((input_size[1] - self.filter_size[1]) // self.stride ) + 1)
        output = np.zeros((output_size[0], output_size[1],input_size[2]))
        for i in range(0,input_size[0] - self.filter_size[0] +1, self.stride):
            for j in range(0, input_size[1] - self.filter_size[1] +1 ,self.stride):
                for n in range(input_size[2]):
                    input_patch = input[i:i+self.filter_size[0], j:j+self.filter_size[1], n]
                    if self.mode == "max":
                        # Max pooling
                        output[i//self.stride][j//self.stride][n] = np.max(input_patch)
                    elif self.mode == "average":
                        # Average pooling
                        output[i//self.stride][j//self.stride][n] = np.mean(input_patch)
        self.output_size = output.shape

        self.output = output
        return output
    
    def backward(self, front_deltas=None, label=None, front_weights=None):
        if self.mode == "max":
            self.weights = np.full(self.filter_size, 1 / (self.filter_size[0] * self.filter_size[1]))

            self.deltas = np.zeros(self.output_size)
            for k in range(self.output_size[2]):
                for l in range(len(front_weights)):
                    self.deltas[:, :, k] += self.fullconv(front_deltas[:, :, l], front_weights[l, :, :, k])
        elif self.mode == "average":
            self.weights = np.full(self.filter_size, 1 / (self.filter_size[0] * self.filter_size[1]))

            self.deltas = np.zeros(self.output_size)
            for k in range(self.output_size[2]):
                for l in range(len(front_weights)):
                    self.deltas[:, :, k] += self.fullconv(front_deltas[:, :, l], front_weights[l, :, :, k])

        return self.deltas
    
    def getModel(self):
        model = {
            "type": f"{self.mode}_pooling2d",
            "params": {}
        }
        return model
    
    def showModel(self):
        print(f"Pooling              {self.output_size}              0")
        print("________________________________________________________")

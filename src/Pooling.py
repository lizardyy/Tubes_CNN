import numpy as np

class Pooling:
    def __init__(self,  filter_size, stride, mode):
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

    def forward(self, input):
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
        return output
    
    def getModel(self):
        model = {
            "type": "max_pooling2d",
            "params": {}
        }
        return model
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
    
    def update_weights(self, learning_rate):
        return

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
    
    def getModel(self):
        model = {
            "type": f"{self.mode}_pooling2d",
            "params": {}
        }
        return model
    
    def summary(self):
        print(f"Pooling              {self.output_size}              0")
        print("________________________________________________________")

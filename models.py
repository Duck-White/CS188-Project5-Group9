from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module
import torch
from torch.nn import Parameter, Linear
from torch import tensor, ones, matmul
from torch.nn.functional import relu, cross_entropy, softmax
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset, LanguageIDDataset

class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()
        "*** YOUR CODE HERE ***"
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        return self.w

    def run(self, x):
        return self(x)

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        return matmul(x, self.w.T)

    def get_prediction(self, x):
        "*** YOUR CODE HERE ***"
        return 1 if self(x).item() >= 0 else -1

class RegressionModel(Module):
    def __init__(self):
        super().__init__()
        "*** YOUR CODE HERE ***"
        self.fc1 = Linear(1, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 1)

    def run(self, x):
        return self(x)

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)

class DigitClassificationModel(Module):
    def __init__(self):
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, output_size)

    def run(self, x):
        return self(x)

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)

class LanguageIDModel(Module):
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        self.hidden_size = 256
        self.wx = Linear(self.num_chars, self.hidden_size)
        self.wh = Linear(self.hidden_size, self.hidden_size)
        self.fc2 = Linear(self.hidden_size, self.hidden_size)
        self.fc_out = Linear(self.hidden_size, len(self.languages))

    def run(self, xs):
        return self(xs)

    def forward(self, xs):
        "*** YOUR CODE HERE ***"
        h = relu(self.wx(xs[0]))
        for x in xs[1:]:
            h = relu(self.wx(x) + self.wh(h))
        h = relu(self.fc2(h))
        return self.fc_out(h)

def Convolve(input: tensor, weight: tensor):
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    patches = input.unfold(0, weight_dimensions[0], 1).unfold(1, weight_dimensions[1], 1)
    Output_Tensor = (patches * weight).sum(dim=(-2, -1))
    return Output_Tensor

class DigitConvolutionalModel(Module):
    def __init__(self):
        super().__init__()
        output_size = 10
        self.convolution_weights = Parameter(ones((3, 3)))
        "*** YOUR CODE HERE ***"
        self.fc1 = Linear(26 * 26, 128)
        self.fc2 = Linear(128, output_size)

    def run(self, x):
        return self(x)

    def forward(self, x):
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(x))
        return self.fc2(x)

class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.layer_size = layer_size

    def forward(self, input):
        B, T, C = input.size()
        "*** YOUR CODE HERE ***"
        q = self.q_layer(input)
        k = self.k_layer(input)
        v = self.v_layer(input)

        scores = matmul(k, q.transpose(-2, -1)) / (self.layer_size ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))[0]
        weights = softmax(scores, dim=-1)
        return matmul(weights, v)

import numpy as np
from collections import OrderedDict


class Linear():
    def __init__(self, in_features, out_features, init):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = np.zeros((self.out_features, 1))
        if init == 2:
            self.weight = np.zeros((self.out_features, self.in_features))
        elif init == 1:
            self.weight = np.random.uniform(-0.1, 0.1, (self.out_features, self.in_features))

    def __call__(self, input_):
        return self.forward(input_)

    def forward(self, input_):
        return np.dot(self.weight, input_) + self.bias


class Sigmoid():
    def __call__(self, input_):
        return self.forward(input_)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input_):
        return self.sigmoid(input_)


class Softmax():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, input_):
        return self.forward(input_)

    def forward(self, input_):
        exp_x = np.exp(input_)
        return exp_x / np.sum(exp_x, axis=0)


class Sequential():
    def __init__(self, *args):
        self.training = None
        self._modules = OrderedDict()
        self.parameters = OrderedDict()
        for name, module in enumerate(args):
            self.add_module(name, module)
            self.register_parameter(name, module)

    def __call__(self, input_):
        return self.forward(input_)

    def register_parameter(self, name, module):  # this suppose to be in Class Module
        if isinstance(module, Linear):
            self.parameters[name] = {'weight':module.weight,
                                     'bias': module.bias}

    def add_module(self, name, module):
        self._modules[name] = module

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)

    def forward(self, input_):
        for name, module in self._modules.items():
            input_ = module(input_)
        return input_

    def output(self, input_):  # record the output result of every layer
        result = [input_]
        for name, module in self._modules.items():
            input_ = module(input_)
            result.append(input_)
        return result

    def update_param(self, params):  # update the parameter in the Linear model using optimizer
        self._modules[0].weight = params[0]['weight']
        self._modules[0].bias = params[0]['bias']
        self._modules[2].weight = params[2]['weight']
        self._modules[2].bias = params[2]['bias']



class optim_SGD():  # suppose bias are combined into weights
    def __init__(self, params, lr):
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")
        self.param_groups = params
        self.g_beta = None  # start from here, followed the HW instruction
        self.g_beta_bias = None
        self.g_alpha = None
        self.g_alpha_bias = None
        self.lr = lr

    def zero_grad(self):  # suppose to be in Class Optimizer
        self.g_beta = np.zeros(self.param_groups[2]['weight'].shape)
        self.g_alpha = np.zeros(self.param_groups[0]['weight'].shape)
        self.g_beta_bias = 0
        self.g_alpha_bias = 0

    def backward(self, layer_output, y_predict, y_true):
        X = layer_output[0]
        Z = layer_output[2]
        dl_db = np.array(y_predict - y_true.reshape(10, 1))
        self.g_beta = np.dot(dl_db, Z.T)
        dl_dz = np.dot(self.param_groups[2]['weight'].T, dl_db)
        dl_da = np.multiply(dl_dz, (np.multiply(Z, 1-Z)))
        self.g_alpha = np.dot(dl_da, X.T)
        self.g_beta_bias = dl_db
        self.g_alpha_bias = dl_da

    def step(self):
        self.param_groups[2]['weight'] = self.param_groups[2]['weight'] - self.lr * self.g_beta
        self.param_groups[0]['weight'] = self.param_groups[0]['weight'] - self.lr * self.g_alpha
        self.param_groups[2]['bias'] = self.param_groups[2]['bias'] - self.lr * self.g_beta_bias
        self.param_groups[0]['bias'] = self.param_groups[0]['bias'] - self.lr * self.g_alpha_bias


def cross_entropy(pred, target, eps=1e-15):
    '''

    :param pred: predicted label: [N * 10]
    :param target: original label: [N * 10]
    :return:
    '''
    loss = -np.sum(np.log(
        np.sum(target * pred, axis=1))) / pred.shape[0]
    return loss

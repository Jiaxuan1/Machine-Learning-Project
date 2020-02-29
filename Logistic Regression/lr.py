

import numpy as np
from math import *
import sys

class LogisticClassifier():
    # initialize the model parameter, initial values and gradient
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.theta = np.zeros(input_dim)
        self.bias = 0.0
        self.dtheta = {}
        self.db = 0.0
        self.train_loss = []
        self.val_loss = []

    def dot_sparse(self, feature_i):
        # calculate the dot product of theta and sparse matrix features
        result = 0
        for feature_idx in feature_i.keys():  # 'idx:value'
            result += self.theta[feature_idx] * feature_i[feature_idx]
        result += self.bias
        return result

    def cross_entropy_loss(self, features, labels):
        result = 0
        for idx, feature_i in enumerate(features):
            result += -(labels[idx] * self.dot_sparse(feature_i)) + log(1 + exp(self.dot_sparse(feature_i)))
        return result / len(features)

    def gradient_reset(self):
        self.dtheta = {}
        self.db = 0.0

    def SGD_optimizer(self, features, labels, lr=0.1):
        # use the i-order of input data, not shuffle
        # random_idx = random.randint(0, len(features)-1)

        for idx, feature_i in enumerate(features):
            self.gradient_reset()
            activation = sigmoid(self.dot_sparse(feature_i))
            for feature_idx in feature_i: # 'idx:value'
                self.dtheta[feature_idx] = feature_i[feature_idx] * (labels[idx] - activation)

            # gradient of the bias, 对应xi=1，故省略
            self.db = labels[idx] - activation

            # update the parameter
            for g in self.dtheta.keys():
                self.theta[g] += lr * self.dtheta[g]
            self.bias += lr * self.db

    def predict(self, features, threshold=0.5):
        predicted_labels = []
        for feature in features:
            activation = sigmoid(self.dot_sparse(feature))
            if activation >= threshold:
                predicted_labels.append(1)
            else: predicted_labels.append(0)
        return predicted_labels

    def calError(self, labels, predicted_labels):
        error_count = [1 for i in range(len(labels)) if labels[i] != predicted_labels[i]]
        return sum(error_count) / len(labels)

def read_dict(dict_input):
    corpus = {}
    with open(dict_input, 'r') as f:
        for line in f:
            line = line.split(' ')
            corpus[line[0]] = int(line[1])
    return corpus

def read_data(path):
    # read the formatted tsv into array
    labels, features = [], []
    with open(path) as f:
        for line in f:
            line = line.strip().split('\t')
            labels.append(int(line[0]))
            features.append({int(idx_value.split(':')[0]):int(idx_value.split(':')[1]) for idx_value in line[1:]})
    return labels, features

def sigmoid(x):
        return 1.0 / (1.0 + exp(-x))

def writeLabel(path, labels):
    with open(path, "w") as f:
        for label in labels:
            f.write(str(label) + "\n")

def writeMatrix(path, train_error, test_error):
    with open(path, "w") as f:
        f.write("error(train): %.6f\n" % train_error)
        f.write("error(test): %.6f\n" % test_error)

def main():
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    # start = time.time()

    train_labels, train_features = read_data(formatted_train_input)
    val_labels, val_features = read_data(formatted_validation_input)
    test_labels, test_features = read_data(formatted_test_input)

    vocab = read_dict(dict_input)
    lr_model = LogisticClassifier(len(vocab))

    # train_loss, val_loss = [], []
    for epoch in range(num_epoch):
        lr_model.SGD_optimizer(train_features, train_labels)
        # train_loss.append(lr_model.cross_entropy_loss(train_features, train_labels))
        # val_loss.append(lr_model.cross_entropy_loss(val_features, val_labels))
    #
    # plt.plot(np.arange(num_epoch), train_loss, c="blue", label="training loss")
    # plt.plot(np.arange(num_epoch), val_loss, c="red", label="validation loss")
    # plt.xlabel("epoch")
    # plt.ylabel("negative log likelihood")
    # plt.title('Model 1')
    # plt.show()

    train_predicted_labels = lr_model.predict(train_features)
    test_predicted_labels = lr_model.predict(test_features)

    train_error = lr_model.calError(train_labels, train_predicted_labels)
    test_error = lr_model.calError(test_labels, test_predicted_labels)

    writeLabel(train_out, train_predicted_labels)
    writeLabel(test_out, test_predicted_labels)
    writeMatrix(metrics_out, train_error, test_error)

    # end = time.time()
    # print('time: %f' % (end-start))

if __name__ == '__main__':
    main()
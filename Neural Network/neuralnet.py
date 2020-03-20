# implement an classifier using a one hidden layer with sigmoid activation and softmax on the output layer from scratch

import numpy as np
import csv
import mytorch as mynn
import sys


def encode(y_arr):
    encoded_target = np.zeros((len(y_arr), 10))
    for row, label in enumerate(y_arr):
        encoded_target[row, label] = 1  # change into one hot encoding
    return encoded_target

def decode(y_arr):
    decode = []
    for row in y_arr:
        decode.append(np.argmax(row))
    return np.array(decode)

def read_data(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader])
    y_label = encode(data[:, 0].astype(int))
    return data[:, 1:].astype(int), y_label

def label_out(path, label):
    with open(path, 'w') as f:
        for i in label:
            f.write(str(i)+'\n')

def metric_out(path, train_CE, test_CE, train_error, test_error):
    with open(path, 'w') as f:
        for i in range(len(train_CE)):
            f.write('epoch=%d crossentropy(train): %f\n' % (i + 1, train_CE[i]))
            f.write('epoch=%d crossentropy(test): %f\n' % (i + 1, test_CE[i]))
        f.write('error(train): %f\n' % train_error)
        f.write('error(test): %f\n' % test_error)

train_input = 'largeTrain.csv'
test_input = 'largeTest.csv'
train_out = 'model1train_out.labels'
test_out = 'model1test_out.labels'
metrics_out = 'model1metrics_out.txt'
num_epoch = 2
hidden_units = 50
init_flag = 1
learning_rate = 0.01
# train_input = sys.argv[1]
# test_input = sys.argv[2]
# train_out = sys.argv[3]
# test_out = sys.argv[4]
# metrics_out = sys.argv[5]
# num_epoch = int(sys.argv[6])
# hidden_units = int(sys.argv[7])
# init_flag = int(sys.argv[8])
# learning_rate = float(sys.argv[9])


train_data, train_label = read_data(train_input)
test_data, test_label = read_data(test_input)

model = mynn.Sequential(
    mynn.Linear(len(train_data[0]), hidden_units, init_flag),
    mynn.Sigmoid(),
    mynn.Linear(hidden_units, 10, init_flag),
    mynn.Softmax(10)
)
optimizer = mynn.optim_SGD(model.parameters, learning_rate)
train_CE = []
test_CE = []
for epoch in range(num_epoch):
    for idx, x in enumerate(train_data):
        x = x.reshape(1, len(train_data[0]))
        layer_output = model.output(x.T)
        y_pred = model(x.T)
        y_true = train_label[idx].reshape(10, 1)
        optimizer.zero_grad()
        optimizer.backward(layer_output, y_pred, y_true)
        optimizer.step()

        model.update_param(optimizer.param_groups)

    train_y_pred = model(train_data.T).T
    train_cross_entropy = mynn.cross_entropy(train_y_pred, train_label)
    test_y_pred = model(test_data.T).T
    test_cross_entropy = mynn.cross_entropy(test_y_pred, test_label)
    train_CE.append(train_cross_entropy)
    test_CE.append(test_cross_entropy)

train_error = np.mean(decode(train_label) != decode(train_y_pred))
test_error = np.mean(decode(test_label) != decode(test_y_pred))

label_out(train_out, decode(train_y_pred))
label_out(test_out, decode(test_y_pred))
metric_out(metrics_out, train_CE, test_CE, train_error, test_error)
print(train_error)
print(test_error)




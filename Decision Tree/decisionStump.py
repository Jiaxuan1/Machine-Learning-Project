# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:27:14 2020

@author: ZJX
"""
import sys 

def readData(path, idx):
    '''
    Read the txt file into a pandas dataframe
    '''
    data = []
    with open(path, 'r') as f:
        title = f.readline().strip().split('\t')
#        data.append((title[idx], title[-1]))
        for line in f:
            record = line.strip().split('\t')
            data.append((record[idx], record[-1]))
    return data

def trainModel(data):
    attribute0 = data[0][0]
    output0 = data[0][1]
    attribute1, output1 = -100, -100
    for record in data:
        if record[0] != attribute0:
            attribute1 = record[0]
        elif record[1] != output0:
            output1 = record[1]
        elif attribute1 != -100 and output1 != -100:
            break
    
    count0 = len([(x, y) for (x, y) in data if x == attribute0 and y == output0])
    count1 = len([(x, y) for (x, y) in data if x == attribute0 and y == output1])
    count2 = len([(x, y) for (x, y) in data if x == attribute1 and y == output0])
    count3 = len([(x, y) for (x, y) in data if x == attribute1 and y == output1])
    result = {attribute0: 0,
              attribute1: 0}
    if count0 > count1:
        result[attribute0] = output0
    elif count0 < count1:
        result[attribute0] = output1
    else:
        result[attribute0] = None
    
    if count2 > count3:
        result[attribute1] = output0
    elif count2 < count3:
        result[attribute1] = output1
    else:
        result[attribute1] = None
    return result

def hypothesis(model, data):
    label = []
    for record in data:
        for key in model:
            if key == record[0]:
                label.append(model[key])
                break
    return label

def calError(train_output, test_output, train_data, test_data):
    train_error_count, test_error_count = 0, 0
    for i in range(len(train_data)):
        if train_output[i] != train_data[i][1]:
            train_error_count += 1
    for i in range(len(test_data)):
        if test_output[i] != test_data[i][1]:
            test_error_count += 1
    return (train_error_count/len(train_data), test_error_count/len(test_data))

def labelOutput(output, out_path):
    with open(out_path, "w") as outputFile:
        for line in output:
            outputFile.write("%s\n" % line)

def metricsOutput(error, out_path):
    with open(out_path, "w") as outputFile:
        for idx, e in enumerate(error):
            if idx == 0:
                outputFile.write("error(train): %f\n" % e)
            elif idx == 1:
                outputFile.write("error(test): %f\n" % e)
    
def main():
#    train_input = 'politicians_train.tsv'
#    test_input = 'politicians_test.tsv'
#    split_index = 3
#    train_out = 'pol_%s_train.labels' % split_index
#    test_out = 'pol_%s_test.labels' % split_index
#    metrics_out = 'pol_%s_metrics.txt' % split_index
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_index = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    
    train_data = readData(train_input, split_index)
    test_data = readData(test_input, split_index)
    
    model = trainModel(train_data)
    
    train_output = hypothesis(model, train_data)
    test_output = hypothesis(model, test_data)
    error = calError(train_output, test_output, train_data, test_data)
    
    labelOutput(train_output, train_out)
    labelOutput(test_output, test_out)
    metricsOutput(error, metrics_out)
    
if __name__ == '__main__':
    main()
    
    
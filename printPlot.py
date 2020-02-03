import decisionTree
import numpy as np
import matplotlib.pyplot as plt

train_input = 'politicians_train.tsv'
test_input = 'politicians_test.tsv'

myTree = decisionTree.binaryDecisionTreeClassifier()

train_features, train_data = myTree.readTSV(train_input)
test_features, test_data = myTree.readTSV(test_input)
max_depth_list = np.arange(0, len(train_features))
train_error, test_error = [], []
for depth in max_depth_list:
    myTree.trainTree(train_features, train_data, depth)
    train_label = myTree.classify(train_features, train_data)
    test_label = myTree.classify(test_features, test_data)
    error = myTree.calError(train_label, test_label, train_data, test_data)
    train_error.append(error[0])
    test_error.append(error[1])
plt.title('Error Change')
plt.xlabel('Max Depth')
plt.ylabel('Error')
plt.plot(max_depth_list, train_error, color='skyblue', label='Training Error')
plt.plot(max_depth_list, test_error, color='red', label='Testing Error')
plt.legend()
plt.savefig('111.png')
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:42:01 2020

@author: ZJX
"""
import sys
import numpy as np

class binaryDecisionTreeClassifier():
    '''
    Binary decision tree classifier: the attributes can be multivalued, the label should be binary
    self.tree: a tree structure to store the training model using dictionary
    self.currentLevel: an int value to determine the current level of decision tree
    self.labels: a list to store the unique labels in this dataset
    '''
    def __init__(self):
        self.tree = {}
        self.currentLevel = 1
        self.labels = []

    def readTSV(self, path):
        '''
        :param path: tsv file path
        :return: features(list), dataset(np.array)
        '''
        with open(path, 'r') as f:
            features = f.readline().strip().split('\t')
            dataset = np.array([line.strip().split('\t') for line in f])
        self.labels = self.drop_duplicate_with_order(list(dataset[:, -1]))
        return features, dataset

    def drop_duplicate_with_order(self, l):
        return sorted(set(l), key=l.index, reverse=True)

    def giniImpurity(self, D):
        '''
        Calculate the gini impurity of the current dataset(aka. node in the tree structure)
        :param D: dataset(np.array)
        :return: gini impurity(float)
        '''
        lbl = list(D[:, -1])
        p_list = [lbl.count(i)/len(lbl) for i in list(set(lbl))]
        return 1 - sum([p**2 for p in p_list])

    def giniGain(self, D, idx):
        '''
        Given a dataset D and the attribute(idx), calculate the gini gain
        :param D: dataset(np.array)
        :param idx: the index of splitting feature
        :return: gini gain
        '''
        unique_value_idx = self.drop_duplicate_with_order(list(D[:, idx]))
        sub_Prob = [list(D[:, idx]).count(i) / len(D[:, idx]) for i in unique_value_idx]
        sub_dataset = [D[D[:, idx] == i] for i in unique_value_idx]
        sub_Gini = [self.giniImpurity(sub) for sub in sub_dataset]
        return self.giniImpurity(D) - sum([sub_Prob[i] * sub_Gini[i] for i, p in enumerate(sub_Prob)])
    
    def splitDataset(self, D, F, idx):
        '''
        Given a dataset and idx, return a dict to contain the subset splited by idx
        :param D: dataset(np.array)
        :param F: features(list)
        :return: splitting dataset {feature: {value1: ..., value2: ..., value n:...}}
        '''
        unique_value_idx = self.drop_duplicate_with_order(list(D[:, idx]))
        sub_d = {F[idx]: {i: D[D[:, idx] == i] for i in unique_value_idx}}
        for value in sub_d[F[idx]]:
            sub_d[F[idx]][value] = np.delete(sub_d[F[idx]][value], idx, axis=1)
        return sub_d

    def majorityVote(self, label):
        '''
        Using majority vote to select the label
        :param label: labels in dataset(list)
        :return: one of the labels
        '''
        unique_label = self.drop_duplicate_with_order(label)
        count_0 = list(label).count(unique_label[0])
        count_1 = list(label).count(unique_label[1])
        if count_0 > count_1:
            return unique_label[0]
        elif count_1 > count_0:
            return unique_label[1]
        else:  # If the vote is tied, choose the attribute to split on that comes last in the lexicographical order
            return sorted(unique_label, reverse=True)[0]

    def trainTree(self, features, dataset, max_depth):
        '''
        Using recursion and dictionary to train the decision tree model
        :param features: list
        :param dataset: np.array
        :param max_depth: int, to avoid overfitting
        :return: the final labels(leaf) or sub-decision tree(node)
        '''
        label_list = list(dataset[:, -1])
        if max_depth == 0:  # Simply using majority vote when max depth is 0
            self.tree = {'majority': self.majorityVote(label_list)}
            return None
        elif len(set(dataset[:, -1])) == 1:  # the labels in the dataset has been pure
            self.currentLevel -= 1
            return list(set(label_list))[0]
        elif len(features) == 1:  # there is zero features, the only column is label
            self.currentLevel -= 1
            return self.majorityVote(label_list)
        elif self.currentLevel > max_depth:  # the levels of current tree is larger than max_depth
            self.currentLevel -= 1
            return self.majorityVote(label_list)

        gini_gain_list = [self.giniGain(dataset, idx) for idx in range(len(features)-1)]
        if max(gini_gain_list) <= 0:  # examine whether the Gini gain is negative
            self.currentLevel -= 1
            return self.majorityVote(list(dataset[:, -1]))
        else:
            select_idx = gini_gain_list.index(max(gini_gain_list))  # suppose we don't have tie in Gini gain list
            select_fea = features[select_idx]
            new_tree, new_tree[select_fea]  = {}, {}

            sub_dataset = self.splitDataset(dataset, features, select_idx)
            sub_features = features.copy()
            sub_features.pop(select_idx)
            for value, subset in sub_dataset[select_fea].items():
                self.currentLevel += 1
                new_tree[select_fea][value] = self.trainTree(sub_features, subset, max_depth)
            self.tree = new_tree
            self.currentLevel -= 1
        return new_tree

    def rowClassify(self, features, row, tree):
        '''
        Given the decision tree and a vector of data, using recursion to output the label
        :param features: list
        :param row: a vector of data (list)
        :param tree: tree nodes in recursion (dictionary)
        :return: label(leaf) or subtree(node)
        '''
        if list(tree.keys())[0] == 'majority':
            return tree['majority']
        current_feature = list(tree.keys())[0]
        idx = features.index(current_feature)
        for key, value in tree[current_feature].items():
            if key == row[idx]:
                if isinstance(value, dict) is False:
                    return tree[current_feature][key]
                else:
                    sub_row = row[:]
                    sub_features = features[:]
                    sub_row.pop(idx)
                    sub_features.pop(idx)
                    return self.rowClassify(sub_features, sub_row, tree[current_feature][key])

    def classify(self, features, dataset):
        '''
        :param features: list
        :param dataset: the whole dataset(np.array)
        :return: output labels (list)
        '''
        return [self.rowClassify(features, list(i), self.tree) for i in dataset]

    def calError(self, train_label, test_label, train_data, test_data):
        '''
        :param train_label: training output(list)
        :param test_label: testing output(list)
        :param train_data: original training data(list)
        :param test_data: original testing data(list)
        :return: training error and testing error(float)
        '''
        train_error_count, test_error_count = 0, 0
        for i in range(len(train_data)):
            if train_label[i] != train_data[i][-1]:
                train_error_count += 1
        for i in range(len(test_data)):
            if test_label[i] != test_data[i][-1]:
                test_error_count += 1
        return train_error_count / len(train_data), test_error_count / len(test_data)

    def labelOutput(self, out_data, out_path):
        with open(out_path, "w") as outputFile:
            for line in out_data:
                outputFile.write("%s\n" % line)

    def metricsOutput(self, error, out_path):
        with open(out_path, "w") as outputFile:
            for idx, e in enumerate(error):
                if idx == 0:
                    outputFile.write("error(train): %f\n" % e)
                elif idx == 1:
                    outputFile.write("error(test): %f\n" % e)

    def resetCurrentLevel(self):
        self.currentLevel = 1

    def printLabels(self, dataset):
        # labels = self.drop_duplicate_with_order(list(dataset[:, -1]))
        count_list = [list(dataset[:, -1]).count(label) for label in self.labels]
        record = '[%d %s/%d %s]' % (count_list[0], self.labels[0], count_list[1], self.labels[1])
        return record

    def printTree(self, tree, features, dataset):
        if list(tree.keys())[0] == 'majority':
            print(self.printLabels(dataset))
            return None
        if self.currentLevel == 0:
            self.currentLevel += 1
            print(self.printLabels(dataset))
        current_feature = list(tree.keys())[0]
        idx = features.index(current_feature)
        sub_dataset = self.splitDataset(dataset, features, idx)
        for key, value in tree[current_feature].items():
            print('| ' * self.currentLevel + '%s = %s: ' % (current_feature, key) + \
                  self.printLabels(sub_dataset[current_feature][key]))
            if isinstance(value, dict):
                sub_features = features[:]
                sub_features.pop(idx)
                sub_sub_dataset = sub_dataset[current_feature][key]
                sub_tree = tree[current_feature][key]
                self.currentLevel += 1
                self.printTree(sub_tree, sub_features, sub_sub_dataset)
        self.currentLevel -= 1


def main():
    train_input = 'politicians_train.tsv'
    test_input = 'politicians_test.tsv'
    max_depth = 3
    train_out = 'pol_%s_train.labels' % max_depth
    test_out = 'pol_%s_test.labels' % max_depth
    metrics_out = 'pol_%s_metrics.txt' % max_depth

    # train_input = sys.argv[1]
    # test_input = sys.argv[2]
    # max_depth = int(sys.argv[3])
    # train_out = sys.argv[4]
    # test_out = sys.argv[5]
    # metrics_out = sys.argv[6]

    myDecisionTree = binaryDecisionTreeClassifier()
    train_features, train_data = myDecisionTree.readTSV(train_input)
    test_features, test_data = myDecisionTree.readTSV(test_input)

    myDecisionTree.trainTree(train_features, train_data, max_depth)

    train_label = myDecisionTree.classify(train_features, train_data)
    test_label = myDecisionTree.classify(test_features, test_data)
    error = myDecisionTree.calError(train_label, test_label, train_data, test_data)

    myDecisionTree.labelOutput(train_label, train_out)
    myDecisionTree.labelOutput(test_label, test_out)
    myDecisionTree.metricsOutput(error, metrics_out)

    myDecisionTree.printTree(myDecisionTree.tree, train_features, train_data)

if __name__ == '__main__':
    main()
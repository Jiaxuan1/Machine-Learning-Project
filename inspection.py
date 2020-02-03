# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:23:18 2020

@author: ZJX
"""
import random
import sys

def readData(path):
    '''
    read the label of tsv file and convert it to 0 and 1
    '''
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        label = []
        for line in f:
            label.append(line.strip().split('\t')[-1])
    variable = list(set(label))
    for i, lbl in enumerate(label):
        if lbl == variable[0]:
            label[i] = 0
        elif lbl == variable[1]:
            label[i] = 1
    return label

def GiniImpurity(lbl):
    P_0 = lbl.count(0) / len(lbl)
    P_1 = lbl.count(1) / len(lbl)
    return 1 - (P_0 ** 2 + P_1 ** 2)

def ErrorRate(lbl):
    if lbl.count(0) > lbl.count(1):
        result = 0
    elif lbl.count(0) < lbl.count(1):
        result = 1
    else:
        result = random.randint(0, 1)
    return 1 - lbl.count(result) / len(lbl)

def outputData(path, gini, error):
    with open(path, 'w') as f:
        f.write('gini_impurity: %f\n' % gini)
        f.write('error: %f' % error)
        

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    label = readData(input_path)
    outputData(output_path, GiniImpurity(label), ErrorRate(label))

if __name__=='__main__':
    main()
    
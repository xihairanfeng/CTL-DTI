import numpy as np
import pandas as pd

'''
处理原始数据集并且统计原始数据集的各个类的边的条数
| origin name        | new name   | edge number |
| ------------------ | ---------- | ----------- |
| drug_drug.txt      | durg_0.txt | 331308      |
| drug_enzyme.txt    | durg_1.txt | 5992        |
| drug_path.txt      | durg_2.txt | 23492       |
| drug_structure.txt | durg_3.txt | 209798      |
| drug_target.txt    | durg_4.txt | 8630        |

'''
import os

def countNodeNumber(edgeArray):
    lenEdge = len(edgeArray)
    print('The edge number : ', lenEdge)
    leftNode = []
    rightNode = []
    # ALLNode = []
    for i in range(lenEdge):
        # print(edgeArray[i][0], edgeArray[i][1])
        # print(edgeArray[i])
        leftNode.append(edgeArray[i][0])
        rightNode.append(edgeArray[i][1])
    ALLNode = leftNode + rightNode
    # print('----------', leftNode)
    # print('----------', ALLNode)
    #统计左节点数
    left_node = list(set(leftNode))
    right_node = list(set(rightNode))
    all_node = list(set(rightNode))
    countDrug = 0
    countOthers = 0

    #Drug的id号小于等于841
    for x in all_node:
        if x < 842 :
            countDrug += 1
        else:
            countOthers += 1
    # print('##############', left_node)
    # print('---------------', right_node)
    print('The Drug node number : ', countDrug)
    print('The Other node number : ', countOthers)
    print('The all node number : ', len(all_node))

filename = r'..\originDataset'

for root, dirs, files in os.walk(filename):
    for i, file in enumerate(files):
        print('--------------------------------------------------------\n'
              '--------------------------------------------------------')
        print('The network layer: ', i, 'The filename is: ', file)
        path = os.path.join(root, file)
        originData = pd.read_csv(path, sep=' ', header=None)
        edgeNumber = len(originData)
        edge = originData.iloc[:, 0:2]
        edgeArray = np.array(edge)
        countNodeNumber(edgeArray)
        print('The edge number in', file, 'is: ', edgeNumber)
        print('********************************************************\n'
              '********************************************************\n')

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

filename = r'..\originDataset'

for root, dirs, files in os.walk(filename):
    for i, file in enumerate(files):
        path = os.path.join(root, file)
        originData = pd.read_csv(path, sep=' ', header=None)
        edgeNumber = len(originData)
        edge = originData.iloc[:, 0:2]
        edgeArray = np.array(edge)
        print('The edge number in', file, 'is: ', edgeNumber)
        newFile_name = '../dataset/durg_' + str(i) + '.txt'
        newFile = open(newFile_name, mode='w')
        for i in edgeArray:
            newFile.write("{} {}\n".format(i[0], i[1]))
        newFile.close()

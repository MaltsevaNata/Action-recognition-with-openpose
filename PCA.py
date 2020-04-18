# Feature Extraction with PCA

import numpy as np
from numpy.linalg import eig
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import csv

classes = ['noseX',	'noseY',	'neckX',	'neckY',	'RShoulderX',	'RShoulderY',	'RElbowX',	'RElbowY',
           'RWristX',	'RWristY',	'LShoulderX',	'LShoulderY',	'LElbowX',	'LElbowY',	'LWristX',	'LWristY',
            'MidHipX',	'MidHipY',	'RHipX',	'RHipY',	'RKneeX',	'RKneeY',	'RAnkleX',	'RAnkleY',	'LHipX',	'LHipY',	'LKneeX',
           'LKneeY',	'LAnkleX',	'LAnkleY',	'REyeX',	'REyeY',	'LEyeX',	'LEyeY',	'REarX',	'REarY',
           'LEarX',	'LEarY',	'LBigToeX',
           'LBigToeY',	'LSmallToeX',	'LSmallToeY',	'LHeelX',	'LHeelY',	'RBigToeX'	,'RBigToeY'	,'RSmallToeX',
           'RSmallToeY',	'RHeelX',	'RHeelY'
]
names = ['move_scaner', 'tune_angle', 'tune_height', 'turn_on']
classes_numbers = [10, 3, 2, 2]
myFile = '/home/natalia/Рабочий стол/Train_data/All/PCA/' + 'all.csv'
newFile = open('/home/natalia/Рабочий стол/Train_data/All/PCA/' + 'myPCA.csv', 'w')
dataframe = read_csv(myFile, header=None)

array = dataframe.values

X = array[:, :50]
M = np.mean(X, axis=0) #среднее каждого столбца
C = X - M #центрировать значения в каждом столбце, вычитая среднее значение столбца.
C = C.transpose()
V = np.cov(C.astype(float)) #матрица ковариации


values, vectors = eig(V)  #список собственных значений и список собственных векторов.
#P = np.dot(values, C.T)
indexes = np.argsort(values)[::-1]
sorted_values = values[indexes]
sorted_vectors = vectors[indexes]
vec = vectors.T[1]
result_pre = np.dot(C.T, sorted_vectors.T[:, 0:20])
#result = result_pre.reshape(-1, 1)

fig, ax = plt.subplots()

ax.bar(np.arange(0,20), result_pre[380])
#labels = classes
#ax.set_xticks(np.arange(len(labels)))
#ax.set_xticklabels(labels, rotation = 90)
ax.set_facecolor('seashell')
fig.set_facecolor('floralwhite')
writer = csv.writer(newFile)
writer.writerows(result_pre)
# Установим логарифмический масштаб по оси X
#ax.set_yscale ('log')

# Установим логарифмический масштаб по оси Y

plt.show()


'''for cl in range(4):
    for number in range(1,classes_numbers[cl]+1):
        dataframe = read_csv('/home/natalia/Рабочий стол/Train_data/All/{}{}.csv'.format(names[cl],number), names=classes)

        array = dataframe.values

        X = array[1:,:]

        shape = (np.shape(X)[0], np.shape(X)[1]+1)
        new_features = np.zeros(shape)
        new_features = new_features.astype(str)
        for i in range(len(X)):
            new_features[i] = np.append(X[i], names[cl])
        writer = csv.writer(myFile)
        writer.writerows(new_features)'''


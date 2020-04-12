# Feature Extraction with PCA

import numpy as np

from pandas import read_csv

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
classes_numbers = [10, 3, 2, 3]
cl = 3
for number in range(1,classes_numbers[cl]+1):
    dataframe = read_csv('/home/natalia/Рабочий стол/Train_data/All/{}{}.csv'.format(names[cl],number), names=classes)
    myFile = open('/home/natalia/Рабочий стол/Train_data/All/PCA/' + '{}{}.csv'.format(names[cl],number), 'w')
    array = dataframe.values

    X = array[1:,:]

    # feature extraction

    pca = PCA(n_components=20)

    fit = pca.fit(X)

    features = fit.transform(X)

    writer = csv.writer(myFile)
    writer.writerows(features)


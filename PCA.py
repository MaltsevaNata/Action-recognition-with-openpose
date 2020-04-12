# Feature Extraction with PCA

import numpy as np

from pandas import read_csv

from sklearn.decomposition import PCA


classes = ['noseX',	'noseY',	'neckX',	'neckY',	'RShoulderX',	'RShoulderY',	'RElbowX',	'RElbowY',
           'RWristX',	'RWristY',	'LShoulderX',	'LShoulderY',	'LElbowX',	'LElbowY',	'LWristX',	'LWristY',
            'MidHipX',	'MidHipY',	'RHipX',	'RHipY',	'RKneeX',	'RKneeY',	'RAnkleX',	'RAnkleY',	'LHipX',	'LHipY',	'LKneeX',
           'LKneeY',	'LAnkleX',	'LAnkleY',	'REyeX',	'REyeY',	'LEyeX',	'LEyeY',	'REarX',	'REarY',
           'LEarX',	'LEarY',	'LBigToeX',
           'LBigToeY',	'LSmallToeX',	'LSmallToeY',	'LHeelX',	'LHeelY',	'RBigToeX'	,'RBigToeY'	,'RSmallToeX',
           'RSmallToeY',	'RHeelX',	'RHeelY'
]

dataframe = read_csv(url, names=names)

array = dataframe.values

X = array[:,0:8]

Y = array[:,8]

# feature extraction

pca = PCA(n_components=3)

fit = pca.fit(X)

features = fit.transform(X)

# summarize components

print("Explained Variance: %s") % fit.explained_variance_ratio_

print(features[0:5,:])
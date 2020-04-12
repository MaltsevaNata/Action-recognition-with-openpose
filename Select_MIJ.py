import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

classes = ['noseX',	'noseY',	'neckX',	'neckY',	'RShoulderX',	'RShoulderY',	'RElbowX',	'RElbowY',
           'RWristX',	'RWristY',	'LShoulderX',	'LShoulderY',	'LElbowX',	'LElbowY',	'LWristX',	'LWristY',
            'MidHipX',	'MidHipY',	'RHipX',	'RHipY',	'RKneeX',	'RKneeY',	'RAnkleX',	'RAnkleY',	'LHipX',	'LHipY',	'LKneeX',
           'LKneeY',	'LAnkleX',	'LAnkleY',	'REyeX',	'REyeY',	'LEyeX',	'LEyeY',	'REarX',	'REarY',
           'LEarX',	'LEarY',	'LBigToeX',
           'LBigToeY',	'LSmallToeX',	'LSmallToeY',	'LHeelX',	'LHeelY',	'RBigToeX'	,'RBigToeY'	,'RSmallToeX',
           'RSmallToeY',	'RHeelX',	'RHeelY'
]

colors = ['b',
'g',
'r',
'c',
'm',

          ]
ind = np.arange(50)
width = 0.2
ax = plt.subplot()

new_sigma_norm_ms = [0]*50
#ms = [[0]*51, [0]*30, [0]*30, [0]*31, [0]*112, [0]*22,[0]*23, [0]*31, [0]*21, [0]*27  ]
#ms = [[0]*192, [0]*162, [0]*37]
#ms = [[0]*200, [0]*112]
ms = [[0]*63, [0]*21, [0]*12]

for act in range(1,4):
    ms[act-1] = pd.read_csv("/home/natalia/Рабочий стол/Train_data/All/turn_on{}.csv".format(act))
    move_scaner_num = len(ms[act-1])

    A = ms[act-1].iloc[: move_scaner_num, 0:50].values.transpose()
    A = np.array(A)
    sigma_ms = np.var(A, axis=1)
    Is_ms = np.argsort(sigma_ms)[::-1]
    sigmas_ms = sigma_ms[Is_ms]
    Jm = 30
    sigma_norm_ms = []
    sum1 = sum(sigmas_ms[:Jm])
    for i in range(Jm):
        sigma_norm_ms.append(sigmas_ms[i]/sum1)
    cut_Is_ms = Is_ms[:Jm]
    new_sigma_norm_ms[act-1]=[0]*50
    for i in range(Jm):
        new_sigma_norm_ms[act-1][cut_Is_ms[i]] = sigma_norm_ms[i]

    ax.bar(ind+act*width, new_sigma_norm_ms[act-1], width, color=colors[act-1])
ax.set_ylabel('Normalized variance')
ax.set_title('Normalized variances histogram for class turn_on')
ax.set_xticks(ind, classes)
ax.set_yticks(np.arange(0, 0.3, 0.1))

labels = classes
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation = 90)
plt.show()


'''ms2 = pd.read_csv("/home/natalia/Рабочий стол/Train_data/All/move_scaner2.csv")
move_scaner_num2 = 30

A = ms2.iloc[: move_scaner_num2, 0:50].values.transpose()
A = np.array(A)
sigma_ms2 = np.var(A, axis=1)
Is_ms2 = np.argsort(sigma_ms2)[::-1]
sigmas_ms2 = sigma_ms2[Is_ms2]
sigma_norm_ms2 = []
sum2 = sum(sigmas_ms2[:Jm])
for i in range(Jm):
    sigma_norm_ms2.append(sigmas_ms2[i]/sum2)
cut_Is_ms2 = Is_ms2[:Jm]
new_sigma_norm_ms2 = [0]*50
for i in range(25):
    new_sigma_norm_ms2[cut_Is_ms2[i]] = sigma_norm_ms2[i]'''



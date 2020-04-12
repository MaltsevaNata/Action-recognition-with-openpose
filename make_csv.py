import sys
import os
import json
import csv

joints = ['noseX',	'noseY',	'neckX',	'neckY',	'RShoulderX',	'RShoulderY',	'RElbowX',	'RElbowY',
           'RWristX',	'RWristY',	'LShoulderX',	'LShoulderY',	'LElbowX',	'LElbowY',	'LWristX',	'LWristY',
            'MidHipX',	'MidHipY',	'RHipX',	'RHipY',	'RKneeX',	'RKneeY',	'RAnkleX',	'RAnkleY',	'LHipX',	'LHipY',	'LKneeX',
           'LKneeY',	'LAnkleX',	'LAnkleY',	'REyeX',	'REyeY',	'LEyeX',	'LEyeY',	'REarX',	'REarY',
           'LEarX',	'LEarY',	'LBigToeX',
           'LBigToeY',	'LSmallToeX',	'LSmallToeY',	'LHeelX',	'LHeelY',	'RBigToeX'	,'RBigToeY'	,'RSmallToeX',
           'RSmallToeY',	'RHeelX',	'RHeelY'
]
classes = ["Move_scaner", "Tune_angle", "Tune_height", "Turn_on"]
classes_numbers = {"Move_scaner":10, "Tune_angle": 3, "Tune_height":2, "Turn_on":2}
path = "/home/natalia/Рабочий стол/Train_data/All/"
for number in range(1,11):
    myFile = open(path+'move_scaner{}.csv'.format(number), 'w')
    dir = '/home/natalia/Рабочий стол/Train_data/Move_scaner{}'.format(number)
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(joints)
        for root, dirs, files in os.walk(dir):
            for file in files:
                with open(dir+'/'+file, 'r') as file:
                    row = []
                    data = json.loads(file.read())['Person0']
                    for part in data:
                        if float(part['Confidence']) >= 0.5:
                            row.append(part['X'])
                            row.append(part['Y'])
                        else:
                            row.append('0.0')
                            row.append('0.0')
                    writer = csv.writer(myFile)
                    writer.writerow(row)

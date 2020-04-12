import sys
import os
import json
import csv


not_needed = ['RAnkle', 'LAnkle', 'Neck', 'RKnee', 'LKnee',

      'REar',
      'LEar',  'LBigToe',  'LSmallToe',  'LHeel', 'RBigToe',  'RSmallToe',  'RHeel',
      'Background'
]
classes = ["Move_scaner", "Tune_angle", "Tune_height", "Turn_on"]
classes_numbers = {"Move_scaner":10, "Tune_angle": 3, "Tune_height":2, "Turn_on":2}
path = "/home/natalia/Рабочий стол/Train_data/All/"
myFile = open(path+'vectors.csv', 'w')
#vectors = []
#i = 0
with myFile:
    for cl in classes:
        number = 1 if cl != "Move_scaner" else 10
        row = []



        for root, dirs, files in os.walk(path + cl):
            files.sort(reverse=False)
            for file in files:

                #vectors.append([])

                if not file.startswith(cl+str(number)):
                    row.append(cl)
                    print(row)
                    writer = csv.writer(myFile)
                    writer.writerow(row)
                    row = []
                    if number == 10:
                        number = 1
                    else:
                        number+=1

                with open(path + cl + '/' + file, 'r') as file:
                    data = json.loads(file.read())['Person0']
                    for part in data:

                        row.append(part['X'])
                        row.append(part['Y'])
                        #vectors[i].append(part['X'])
                        #vectors[i].append(part['Y'])
                if len(row) > 800:
                    row.append(cl)
                    writer = csv.writer(myFile)
                    writer.writerow(row)
                    row = []


                #i+=1
#with open(path+'vectors.txt', 'w') as file:
    #file.write(str(vectors))


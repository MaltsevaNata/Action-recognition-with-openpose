
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split

import os

# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    'Nose','Neck', 'RShoulder','RElbow', 'RWrist','LShoulder', 'LElbow','LWrist',
     'MidHip',
     'RHip',  'RKnee',  'RAnkle',  'LHip', 'LKnee',  'LAnkle',  'REye',  'LEye',  'REar',
      'LEar',  'LBigToe',  'LSmallToe',  'LHeel', 'RBigToe',  'RSmallToe',  'RHeel',
      'Background'
]

# Output classes to learn how to classify
LABELS = [
    "MOVE_SCANER",
    "TUNE_ANGLE",
    "TUNE_HEIGHT",
    "TURN_ON",
]


df = pd.read_csv("/home/natalia/Рабочий стол/Train_data/All/words.csv")

some = df.sample(10) #набор случайных 30 строк
print(some)
x = df.iloc[:, 0:49].values
y = df.iloc[:, 50].values
#x_new = SelectKBest(f_classif, k=30).fit_transform(x,y)
x_new = x
kmean4 = KMeans(n_clusters=4)
y_kmeans4 = kmean4.fit_predict(x)
print(y_kmeans4)
centers = kmean4.cluster_centers_
print(centers)
labels = kmean4.labels_
print(labels)

vectors = pd.read_csv("/home/natalia/Рабочий стол/Train_data/All/vectors.csv")
train, test = train_test_split(vectors, test_size=0.2)


def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances


bags = {}
bag = 0
for example in train.values:
    cl = None
    index = 850
    while cl == float("nan") or cl == None or pd.isnull(cl):
        cl = example[index]
        index -=1
    if cl in bags:
        bag = len(bags[cl])
        bags[cl].append([])
    else:
        bag = 0
        bags[cl] =[]
        bags[cl].append([])
    element = 0
    while element< (example.size - 50):
        distances = []
        for center in centers:
            pose = example[element:element+49]
            try:
                for el in range(len(pose)):
                    pose[el] = float(pose[el])
            except:
                continue
            dist = np.linalg.norm(pose - center)

            distances.append(dist)
        if distances:
            distance = min(distances)
            cluster = distances.index(distance)

            bags[cl][bag].append(cluster)

        element +=50

print(bags)
'''x1 = bags['Move_scaner']
y1 = np.ones(len(x1))
x2 = bags['Tune_angle']
y2 = np.full(len(x2), 2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('Move_scaner')
ax1.scatter(x1,y1,c=[1, 2, 3, 4],cmap='red')
ax2.scatter(x2, y2,  c='blue')
ax2.set_title("Turn_angle")'''

plt.show()
'''Error =[]
for i in range(1, 25):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 25), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()'''


plt.scatter(x[:,8], x[:,9], c=y_kmeans4, cmap='Dark2')
plt.scatter(centers[:, 0], centers[:, 1],  c='red')
plt.show()
'''classes = []
for cl in y:
    if cl == 'Move_scaner':
        classes.append(1)
    elif cl == 'Tune_angle':
        classes.append(2)
    elif cl == 'Tune_height':
        classes.append(3)
    else:
        classes.append(4)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(x_new[:,0],x_new[:,1],c=y_kmeans4,cmap='brg')
ax1.scatter(centers[:, 0], centers[:, 1],  c='red')
ax2.set_title("Original")
ax2.scatter(x[:,0],x[:,1],c=classes,cmap='brg')
plt.show()'''

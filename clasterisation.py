import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/natalia/Рабочий стол/Train_data/All/PCA/all.csv')


x = df.iloc[:, 0:19].values
y = df.iloc[:, 20].values
#x_new = SelectKBest(f_classif, k=30).fit_transform(x,y)
x_new = x
kmean4 = KMeans(n_clusters=4)
y_kmeans4 = kmean4.fit_predict(x)
print(y_kmeans4)
centers = kmean4.cluster_centers_
print(centers)
labels = kmean4.labels_
print(labels)

classes = []
for cl in y:
    if cl == 'move_scaner':
        classes.append(0)
    elif cl == 'tune_angle':
        classes.append(1)
    elif cl == 'tune_height':
        classes.append(2)
    else:
        classes.append(3)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(x_new[:,0],x_new[:,1],c=y_kmeans4,cmap='brg')
ax1.scatter(centers[:, 0], centers[:, 1],  c='red')
ax2.set_title("Original")
ax2.scatter(x[:,0],x[:,1],c=classes,cmap='brg')
plt.show()
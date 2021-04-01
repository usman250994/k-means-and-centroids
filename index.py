import csv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x = []
with open('data-set-activation-values.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x.append([row[0], row[1]])

x = np.array(x).astype(np.float)

kmeans = KMeans(n_clusters=3, random_state=0)
label = kmeans.fit_predict(x)
u_labels = np.unique(label)

centroids = kmeans.cluster_centers_
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(x[label == i, 0], x[label == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='yellow')
plt.legend()
plt.show()

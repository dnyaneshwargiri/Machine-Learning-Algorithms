from sklearn.cluster import KMeans
import numpy as np

X = np.array([[0.1, 0.6], [0.15, 0.71], [0.08, 0.9],[0.16, 0.85], [0.2, 0.3], [0.25, 0.5],[0.24,0.1],[0.3,0.2]])
#c1=p1 c2=p8
kmeans = KMeans(n_clusters=2)
kmeans.cluster_centers_=[[0.1,0.6],[0.3,0.2]]
print('Initial Centroids are P1 and P8  :',kmeans.cluster_centers_ )
kmeans.fit(X)
print("Cluster labels Pointwise :",kmeans.labels_)
print("P6 belnogs to : ",kmeans.predict([[0.25, 0.5]]))
c1=[]
c2=[]
cnt=0
for i in kmeans.labels_:
    if i==1:
        c1.append(X[cnt])
        cnt+=1
    else:
        c2.append(X[cnt])
        cnt+=1
print('Population around M2 ')
for i in c2:
    print(i)
print('Final Centroids are :',kmeans.cluster_centers_)

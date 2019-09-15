import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier

 #Read dataset 
dataset = pd.read_csv('Kdata.csv')
print (dataset)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,2].values
cluster2=dataset.loc[dataset['Class']=='positive']
cluster1=dataset.loc[dataset['Class']!='positive']
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,y)
X_test=np.array([6,6])
y_pred=classifier.predict([X_test])
print ('general KNN:',y_pred)
classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(X,y)
#predict class for the points (6,6)
X_test=np.array([6,6])
y_pred=classifier.predict([X_test])
print ('Distance weighted KNN:',y_pred)
#you can observe that the output for both the KNN (general and weighted ) is same \n",
#but if you change the points to (6,2) the value for general KNN will be negative and for weighted KNN it will be positive "
print(cluster1.iloc[:,0].values)
print(cluster1.iloc[:,1].values)
print(cluster2.iloc[:,0].values)
print(cluster2.iloc[:,1].values)
plt.scatter(cluster1.iloc[:,0].values,cluster1.iloc[:,1].values,color="red")
plt.scatter(cluster2.iloc[:,0].values,cluster2.iloc[:,1].values,color="yellow")
plt.scatter(6,6,marker="*")
plt.show()


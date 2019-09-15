import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("stock_dataset.csv")
X = df.iloc[0:20,0:7]
Y = df.iloc[0:20,8]
'''print(X)
print(Y)
'''



x_std = StandardScaler().fit_transform(X)
print('Standardized Matrix : \n',x_std)
pca = PCA(n_components=2)
pca.fit(x_std)
x_pca = pd.DataFrame(pca.transform(x_std))
x_pca = x_pca.rename(index=str, columns={0: "PCA1", 1: "PCA2"})
print("\n\nPCA Variance ratio :\n",pca.explained_variance_ratio_)
print("\n\nPCA Single Values :\n",pca.singular_values_)
print("\n\nOriginal shape:   ", x_std.shape)
print("\n\nTransformed shape:", x_pca.shape)
x_pca.insert(2, "BuyorNot",(list)(Y), True)
print('\n\n',x_pca)

'''
plt.scatter(X[0:20,0:7], X[0:20,8], alpha=0.2)
plt.scatter(X_new[0:20,0:7], X_new[0:20,8], alpha=0.8)
plt.axis('equal');
plt.title('PCA result')


plt.show()
'''
sns.relplot(x="PCA1", y="PCA2", hue="BuyorNot", data=x_pca);
plt.show()

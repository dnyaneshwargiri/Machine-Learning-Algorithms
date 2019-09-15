import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing



df = pd.read_csv('cosmetics.csv')
df.drop(columns=['ID'],inplace= True)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
df = pd.DataFrame(df,columns=['Age','Income','Gender','Marital Status'])
le = preprocessing.LabelEncoder()
df=df.apply(le.fit_transform)

print(df)

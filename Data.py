import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from array import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import collections

Rh1 = np.random.uniform(90,100,100)
Rh2 = np.random.uniform(92,100,100)
Rh3 = np.random.uniform(0,50,100)
Rh4 = np.random.uniform(70,100,100)
Rh5 = np.random.uniform(77,85,100)
Rh6 = np.random.uniform(0,50,100)
T1 = np.random.normal(20,1/3,100)
T2 = np.random.normal(28,1/3,100)
T3 = np.random.uniform(20,30,100)
T4 = np.random.uniform(22,30,100)
T5 = np.random.uniform(22.5,23.5,100)
T6 = np.random.uniform(25,45,100)

Stage = ['Seedling', 'Stem Elongation', 'Branching', 'Flowering', 'Fruiting']
col = ['Rh','T','Stage','Disease']

temp=[]
k=0
for i in range(len(Rh1)):
    for j in range(len(T1)):
        temp.insert(k,[Rh1[i],T1[j],'Seedling','Damping Off'])
        k=k+1
df1=pd.DataFrame(temp,columns=col)
temp.clear()
k=0
for i in range(len(Rh2)):
    for j in range(len(T2)):
        temp.insert(k,[Rh2[i],T2[j],'Fruiting','Fruit Rot and Die Back'])
        k=k+1
df2=pd.DataFrame(temp,columns=col)
temp.clear()
k=0
for i in range(len(Rh3)):
    for j in range(len(T3)):
        for s in Stage:
            temp.insert(k,[Rh3[i],T3[j], s, 'Powdery Mildew'])
            k=k+1
df3=pd.DataFrame(temp,columns=col)
temp.clear()
k=0
for i in range(len(Rh4)):
    for j in range(len(T4)):
        for s in ['Branching','Stem Elongation']:
            temp.insert(k,[Rh4[i],T4[j], s, 'Bacterial Leaf Spot'])
            k=k+1
df4=pd.DataFrame(temp,columns=col)
temp.clear()
k=0
for i in range(len(Rh5)):
    for j in range(len(T5)):
        for s in Stage:
            temp.insert(k,[Rh5[i],T5[j], s, 'Cercospora Leaf Spot'])
            k=k+1
df5=pd.DataFrame(temp,columns=col)
temp.clear()
k=0
for i in range(len(Rh6)):
    for j in range(len(T6)):
        for s in Stage:
            temp.insert(k,[Rh6[i],T6[j], s, 'Fusarium Wilt'])
            k=k+1
df6=pd.DataFrame(temp,columns=col)
temp.clear()
k=0

df = df1.append(df2.append(df3.append(df4.append(df5.append(df6, ignore_index=True),ignore_index=True),ignore_index=True),ignore_index=True),ignore_index=True)
df_d = pd.get_dummies(df['Stage'])
df = df.drop(['Stage'], axis=1)
df = pd.concat([df, df_d],axis=1)
file_name = 'data.csv'
df.to_csv(file_name, sep='\t')

features = ['T','Rh']
x = df.loc[:,features].values
y = df.loc[:,'Disease'].values
# scaler = MinMaxScaler(feature_range=(0,1))
# x = scaler.fit_transform(x)
# x = pd.concat([pd.DataFrame(data=x, columns=['T','Rh']), pd.DataFrame(data=df.loc[:,Stage],columns=Stage)], axis=1)
# pca = PCA(n_components=2)
# pC = pca.fit_transform(x)
# x = pd.DataFrame(data=pC, columns=['Axis1','Axis2'])
x = pd.DataFrame(data=x, columns=['T','Rh'])
Df = pd.concat([x,df[Stage],df[['Disease']]],axis=1)

dict={}
index=0
for i in df.loc[:,'Disease'].unique():
    dict[i]=index
    index=index+1
print(dict)
for i in range(len(y)):
    for key in dict.keys():
        if(y[i]==key):
            y[i]=dict[key]

features = features + Stage
print(features)
x = Df.loc[:,features].values
#x = csr_matrix(x, dtype = 'float64')
x , y = shuffle(x,y)
x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.2)

dict1 = collections.Counter(y_tr)
dict2 = collections.Counter(y_te)
dict3 = collections.Counter(y)
print(dict1)
print(dict2)
print(dict3)

clf1 = DecisionTreeClassifier(max_depth=4)
clf1 = clf1.fit(x_tr,y_tr.astype('int'))
r = export_text(clf1)
print(r)
print(clf1.score(x_te,y_te.astype('int')))
print(clf1.score(x_tr,y_tr.astype('int')))
print(clf1.predict_proba(x_te[10:,:]))
print(confusion_matrix(clf1.predict(x_te),y_te.astype('int')))
print(clf1.feature_importances_)

list = ['Damping Off', 'Fruit Rot and Die Back', 'Powdery Mildew', 'Bacterial Leaf Spot', 'Cercospora Leaf Spot', 'Fusarium Wilt' ]
colors = 'rkbycm'
for i, color in zip(clf1.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c=color, edgecolor='black', cmap=plt.cm.Paired, s=20)

plt.xlabel('T')
plt.ylabel('RH')
plt.title('Variation of Diseases with Temperature and Relative Humidity')
plt.legend(list)
plt.show()

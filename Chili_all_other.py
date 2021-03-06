def Chilli_all_other():

  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from array import *
  from sklearn.model_selection import train_test_split
  from sklearn.utils import shuffle
  from sklearn.preprocessing import StandardScaler
  from sklearn.svm import SVC
  from sklearn.metrics import confusion_matrix
  import collections
  import itertools

  Rh3 = np.arange(25,50,0.1)
  Rh4 = np.arange(70,100,0.1)
  Rh5 = np.arange(77,85,0.1)
  Rh6 = np.arange(85,100,0.1)
  T3 = np.arange(20,30,0.1)
  T4 = np.arange(22,30,0.1)
  T5 = np.arange(22,25,0.1)
  T6 = np.arange(25,32,0.1)

  dict = { 0:'Damping Off', 1:'Fruit Rot and Die Back', 2:'Powdery Mildew', 3:'Bacterial Leaf Spot', 4:'Cercospora Leaf Spot', 5:'Fusarium Wilt'}
  Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

  df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
  print(df3.shape)
  df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T',  'Disease'])
  print(df4.shape)
  df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T', 'Disease'])
  print(df5.shape)
  df6 = pd.DataFrame(data=(list(itertools.product(Rh6,T6,[5]))),columns=['Rh', 'T',  'Disease'])
  print(df6.shape)

  df = df3.append(df4.append(df5.append(df6, ignore_index=True),ignore_index=True),ignore_index=True)

  features = ['Rh','T']
  df = df.sample(frac=1).reset_index(drop = True)

  x = df.loc[:,features].values
  y = df.loc[:,['Disease']].values

  l1 = [ 'Powdery Mildew', 'Bacterial Leaf Spot', 'Cercospora Leaf Spot', 'Fusarium Wilt' ]
  colors = 'krby'
  l = [2,3,4,5]
  for i, color in zip(l, colors):
      idx = np.where(y == i)
      plt.scatter(x[idx, 0], x[idx, 1], c=color, edgecolor='black', cmap=plt.cm.Paired, s=20)


  plt.xlabel('Rh')
  plt.ylabel('T')
  plt.title('Variation of Diseases with Temperature and Relative Humidity')
  plt.legend(l1)

  x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
  x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

  clf1 = SVC(probability=True).fit(x_tr,y_tr)
  print(clf1.score(x_tr,y_tr.astype('int')))
  print(clf1.score(x_de,y_de.astype('int')))
  print(clf1.score(x_te,y_te.astype('int')))

  return clf1

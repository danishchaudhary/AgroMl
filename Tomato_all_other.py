def Tomato_all_others():
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from array import *
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix
    import collections
    import itertools

    Rh2 = np.arange(75,100,0.1)
    Rh3 = np.arange(75,100,0.1)
    Rh4 = np.arange(70,100,0.1)
    Rh5 = np.arange(80,100,0.1)
    T2 = np.arange(20,25,0.1)
    T3 = np.arange(25,30,0.1)
    T4 = np.arange(25,30,0.1)
    T5 = np.arange(15,21,0.1)

    dict = { 0:'Damping Off', 1:'Septorial Leaf Spot', 2:'Bacterial Stem and Fruit Canker', 3:'Early Blight', 4:'Bacterial Leaf Spot'}
    Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

    df2 = pd.DataFrame(data=(list(itertools.product(Rh2,T2,[1]))),columns=['Rh', 'T',  'Disease'])
    print(df2.shape)
    df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,[2]))),columns=['Rh', 'T',  'Disease'])
    print(df3.shape)
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T', 'Disease'])
    print(df4.shape)
    df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,[4]))),columns=['Rh', 'T',  'Disease'])
    print(df5.shape)

    df = df2.append(df3.append(df4.append(df5, ignore_index=True),ignore_index=True),ignore_index=True)

    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values

    l1 = [ 'Septorial Leaf Spot', 'Bacterial Stem and Fruit Canker', 'Early Blight', 'Bacterial Leaf Spot' ]
    colors = 'krby'
    l = [1,2,3,4]
    for i, color in zip(l, colors):
        idx = np.where(y == i)
        plt.scatter(x[idx, 0], x[idx, 1], c=color, edgecolor='black', cmap=plt.cm.Paired, s=20)


    plt.xlabel('Rh')
    plt.ylabel('T')
    plt.title('Variation of Diseases with Temperature and Relative Humidity')
    plt.legend(l1)

    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1)
    x_de, x_te, y_de, y_te = train_test_split(x_te,y_te, test_size=0.5)

    clf1 = SVC(kernel='poly', probability=True).fit(x_tr,y_tr)
    print(clf1.score(x_tr,y_tr.astype('int')))
    print(clf1.score(x_de,y_de.astype('int')))
    print(clf1.score(x_te,y_te.astype('int')))

    return clf1

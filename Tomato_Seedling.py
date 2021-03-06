def Tomato_Seedling():
    
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

    Rh1 = np.arange(85,100,0.1)
    Rh4 = np.arange(70,100,0.1)
    T1 = np.arange(18,25,0.1)
    T4 = np.arange(25,30,0.1)

    dict = { 0:'Damping Off', 1:'Septorial Leaf Spot', 2:'Bacterial Stem and Fruit Canker', 3:'Early Blight', 4:'Bacterial Leaf Spot'}
    Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']

    df1 = pd.DataFrame(data=(list(itertools.product(Rh1,T1,[0]))),columns=['Rh', 'T',  'Disease'])
    print(df1.shape)
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,[3]))),columns=['Rh', 'T', 'Disease'])
    print(df4.shape)

    df = df1.append(df4, ignore_index=True)

    features = ['Rh','T']
    df = df.sample(frac=1).reset_index(drop = True)

    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values

    l1 = [ 'Damping Off','Early Blight']
    colors = 'rb'
    l = [0,3]
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

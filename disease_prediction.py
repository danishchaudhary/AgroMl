def disease_prediction(list):
    Rh1 = np.arange(90.1,100,0.1)
    # np.random.uniform(90,100,500)
    Rh2 = np.arange(92.1,100,0.1)
    # np.random.uniform(92,100,500)
    Rh3 = np.arange(0.1,50,0.1)
    # np.random.uniform(0,50,500)
    Rh4 = np.arange(70.1,100,0.1)
    # np.random.uniform(70,100,500)
    Rh5 = np.arange(77.1,85,0.1)
    # np.random.uniform(77,85,500)
    Rh6 = np.arange(0.1,50,0.1)
    # np.random.uniform(0,50,500)
    T1 = np.arange(20.1,28,0.1)
    T2 = np.arange(28.1,32,0.1)
    T3 = np.arange(20.1,30,0.1)
    T4 = np.arange(22.1,30,0.1)
    T5 = np.arange(22.6,23,0.1)
    T6 = np.arange(25.1,45,0.1)

    Stage = ['Branching', 'Flowering', 'Fruiting','Seedling', 'Stem Elongation']
    dict = { 0:'Damping Off', 1:'Fruit Rot and Die Back', 2:'Powdery Mildew', 3:'Bacterial Leaf Spot', 4:'Cercospora Leaf Spot', 5:'Fusarium Wilt'}

    df1 = pd.DataFrame(data=(list(itertools.product(Rh1,T1,['Seedling'],[0]))),columns=['Rh', 'T',  'Stage', 'Disease'])
    # print(df1.shape)
    df2 = pd.DataFrame(data=(list(itertools.product(Rh2,T2, ['Flowering'],[1]))),columns=['Rh', 'T', 'Stage', 'Disease'])
    # print(df2.shape)
    df3 = pd.DataFrame(data=(list(itertools.product(Rh3,T3,Stage,[2]))),columns=['Rh', 'T',  'Stage', 'Disease'])
    # print(df3.shape)
    df4 = pd.DataFrame(data=(list(itertools.product(Rh4,T4,Stage[1:5],[3]))),columns=['Rh', 'T',  'Stage', 'Disease'])
    # print(df4.shape)
    df5 = pd.DataFrame(data=(list(itertools.product(Rh5,T5,Stage,[4]))),columns=['Rh', 'T', 'Stage', 'Disease'])
    # print(df5.shape)
    df6 = pd.DataFrame(data=(list(itertools.product(Rh6,T6,Stage,[5]))),columns=['Rh', 'T', 'Stage', 'Disease'])
    # print(df6.shape)


    df = df1.append(df2.append(df3.append(df4.append(df5.append(df6, ignore_index=True),ignore_index=True),ignore_index=True),ignore_index=True),ignore_index=True)
    df_d = pd.get_dummies(df['Stage'])
    df = df.drop(['Stage'], axis=1)
    df = pd.concat([df, df_d],axis=1)
    file_name = 'data.csv'
    df.to_csv(file_name, sep='\t')

    features = ['Rh','T']
    # x = df.loc[:,features].values
    # y = df.loc[:,'Disease'].values
    # # # scaler = MinMaxScaler(feature_range=(0,1))
    # # # x = scaler.fit_transform(x)
    # # # x = pd.concat([pd.DataFrame(data=x, columns=['T','Rh']), pd.DataFrame(data=df.loc[:,Stage],columns=Stage)], axis=1)
    # # # pca = PCA(n_components=2)
    # # # pC = pca.fit_transform(x)
    # # # x = pd.DataFrame(data=pC, columns=['Axis1','Axis2'])
    # x = pd.DataFrame(data=x, columns=['T','Rh'])
    # Df = pd.concat([x,df[Stage],df[['Disease']]],axis=1)

    df = df.sample(frac=1).reset_index(drop = True)
    # print(df)
    features = features + Stage
    print(features)
    x = df.loc[:,features].values
    y = df.loc[:,['Disease']].values


    # x = csr_matrix(x, dtype = 'float64')
    # x , y = shuffle(x,y)
    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.2)
    #
    # dict1 = collections.Counter(y_tr)
    # dict2 = collections.Counter(y_te)
    # dict3 = collections.Counter(y)
    # print(dict1)
    # print(dict2)
    # print(dict3)

    clf1 = DecisionTreeClassifier(max_depth=4)
    clf1 = clf1.fit(x_tr,y_tr.astype('int'))
    # print(clf1.score(x_te,y_te.astype('int')))
    # print(clf1.score(x_tr,y_tr.astype('int')))
    # print(clf1.predict_proba(x_te[0:10,:]))
    # print(x_te[0:10,:])
    # print(confusion_matrix(clf1.predict(x_te),y_te.astype('int')))
    # print(clf1.feature_importances_)
    #
    # list = ['Damping Off', 'Fruit Rot and Die Back', 'Powdery Mildew', 'Bacterial Leaf Spot', 'Cercospora Leaf Spot', 'Fusarium Wilt' ]
    # colors = 'rkbycm'
    # for i, color in zip(clf1.classes_, colors):
    #     idx = np.where(y == i)
    #     plt.scatter(x[idx, 0], x[idx, 1], c=color, edgecolor='black', cmap=plt.cm.Paired, s=20)
    #
    # plt.xlabel('T')
    # plt.ylabel('RH')
    # plt.title('Variation of Diseases with Temperature and Relative Humidity')
    # plt.legend(list)


    dict_s = {}
    dict_s['Seedling']=[0,0,0,1,0]
    dict_s['Stem Elongation']=[0,0,0,0,1]
    dict_s['Branching']=[1,0,0,0,0]
    dict_s['Flowering']=[0,1,0,0,0]
    dict_s['Fruiting']=[0,0,1,0,0]

    result = []
    i=0
    # for s in Stage:
    #     list = [[74.5,26.2,s],[74.67,26.4,s],[73.96,26.6,s],[74.21,26.4,s],[73.21,26.4,s],[73.21,26.6,s],[75.29,26.2,s]]
    for var in list:
        l=dict_s[var[2]]
        var.pop(2)
        var = var + l
        result.insert(i,clf1.predict_proba(np.asarray(var).reshape(1,-1)))   
    return (result,dict)

    # plt.show()

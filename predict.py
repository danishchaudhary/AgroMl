def predict(Crop, Stage, l):

    if(Crop=='Chilli'):
        if(Stage=='Seedling'):
            for var in l:
                temp.insert(i,chilli_seedling.predict_proba(np.asarray(var).reshape(1,-1)))
                i=i+1
            temp1=[]
            i=0
            for var in temp:
              var=var[0].tolist()
              var=[var[0]]+[0]+var[1:]
              temp1.insert(i,var)
              i=i+1
            return (temp1)

        elif (Stage=='Flowering'):
            for var in l:
                temp.insert(i,chilli_flowering.predict_proba(np.asarray(var).reshape(1,-1)))
                i=i+1
            temp1=[]
            i=0
            for var in temp:
              var=var[0].tolist()
              var=[0]+var
              temp1.insert(i,var)
              i=i+1
            return (temp1)

        else:
            for var in l:
                temp.insert(i,chilli_all_other.predict_proba(np.asarray(var).reshape(1,-1)))
                i=i+1
            temp1=[]
            i=0
            for var in temp:
              var=var[0].tolist()
              var=[0,0]+var
              temp1.insert(i,var)
              i=i+1
            return (temp1)


    if(Crop=='Tomato'):
        if(Stage=='Seedling'):
            for var in l:
                temp.insert(i,tomato_seedling.predict_proba(np.asarray(var).reshape(1,-1)))
                i=i+1
            temp1=[]
            i=0
            for var in temp:
              var=var[0].tolist()
              var=[var[0]]+[0,0]+[var[1]]+[0]
              temp1.insert(i,var)
              i=i+1
            return (temp1)

        else:
            for var in l:
                temp.insert(i,tomato_all_other.predict_proba(np.asarray(var).reshape(1,-1)))
                i=i+1
            temp1=[]
            i=0
            for var in temp:
              var=var[0].tolist()
              var=[0]+var
              temp1.insert(i,var)
              i=i+1
            return (temp1)
        
    if(Crop=='Cotton'):
        if(Stage=='Seedling'):
            for var in l:
                temp.insert(i,cotton_seedling.predict_proba(np.asarray(var).reshape(1,-1)))
                i=i+1
            temp1=[]
            i=0
            for var in temp:
              var=var[0].tolist()
              var=[var[0],var[1]]+[0]+var[2:]
              temp1.insert(i,var)
              i=i+1
            return (temp1)

        elif (Stage=='Flowering'):
            for var in l:
                temp.insert(i,cotton_flowering.predict_proba(np.asarray(var).reshape(1,-1)))
                i=i+1
                temp1=[]    
                i=0
                for var in temp:
                  var=var[0].tolist()
                  var=[var[0]]+[0]+var[1:]
                  temp1.insert(i,var)
                  i=i+1
                temp1  
            return (temp1)

        else:
            for var in l:
            temp.insert(i,cotton_all_other.predict_proba(np.asarray(var).reshape(1,-1)))
            i=i+1
            temp1=[]    
            i=0
            for var in temp:
              var=var[0].tolist()
              var=[var[0]]+[0]+[var[1]]+[0]+[var[2]]
              temp1.insert(i,var)
              i=i+1  
            return (temp1)

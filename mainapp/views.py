from django.shortcuts import render,HttpResponse
from pandas.io.parquet import FastParquetImpl
from scipy.sparse import base
from . import views 
import pandas as pd
from django.shortcuts import render,HttpResponse
from mainapp import functions
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Create your views here.
def basefunc(request):
    if request.method == 'POST':
        file = request.FILES["myfile"]
        options = int(request.POST["dropdown"])
        print(options)
        if options == 1:
            dataset = pd.read_csv(file)
            heads = list(dataset.columns)
            iv = heads[0]
            dv = heads[1]
            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, 1]
            # Splitting the dataset into the Training set and Test set
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
            # Training the Simple Linear Regression model on the Training set\
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            joblib.dump(regressor,'regression_model')
            # newobj = regressor
            score = int(100*(regressor.score(X,y)))
            plt.scatter(X_test, y_test, color = 'red')
            plt.plot(X_train, regressor.predict(X_train), color = 'blue')
            plt.title('Prediction(Test Set)')
            plt.xlabel(iv)
            plt.ylabel(dv)
            plt.savefig('static/plots/SLR/SLR.png',orientation='landscape')
            return render(request, "design.html", {'something':True, 'sm':score})
        elif options == 2:
            dataset = pd.read_csv(file)
            heads = list(dataset.columns)
            iv = heads[0]
            iv1 = heads[1]
            X = dataset.iloc[:,[0,1]]
            y = dataset.iloc[:, -1]
            # Splitting the dataset into the Training set and Test set
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(random_state = 0)
            classifier.fit(X_train, y_train)
            joblib.dump(classifier,'Classification_model')   
            y_pred = classifier.predict(X_test) 
            from sklearn.metrics import accuracy_score
            score = int(100*(accuracy_score(y_test,y_pred)))
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            total =0
            newlst= []
            for i in cm:
                for j in i:
                    total += j
                    newlst.append(j)
            wd = newlst[1]+newlst[2]
            cp = total - wd
            # Visualising the Test set results
            from matplotlib.colors import ListedColormap
            X_set, y_set = X_test, y_test
            title = "Confusion Matrix"
            from sklearn.metrics import plot_confusion_matrix
            disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                 normalize=None)
            disp.ax_.set_title(title)
            plt.savefig('static/plots/SLR/LR.png',orientation='landscape')
            return render(request, "design.html", {'something2':True, 'wd':wd, 'cp':cp,'total':total,'score':score})
        elif options == 3:
                dataset = pd.read_csv(file)
                heads = list(dataset.columns)
                iv = heads[0]
                iv1 = heads[1]
                X = dataset.iloc[:,[0,1]]
                y = dataset.iloc[:, -1]
                # Splitting the dataset into the Training set and Test set
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                #Selecting the model 
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy', random_state = 0)
                classifier.fit(X_train, y_train)
                joblib.dump(classifier,'Randomforest_model')
                y_pred = classifier.predict(X_test) 
                from sklearn.metrics import accuracy_score
                score = int(100*(accuracy_score(y_test,y_pred)))
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                total =0
                newlst= []
                for i in cm:
                    for j in i:
                        total += j
                        newlst.append(j)
                wd = newlst[1]+newlst[2]
                cp = total - wd
                # Visualising the Test set results
                from matplotlib.colors import ListedColormap
                X_set, y_set = X_test, y_test
                title = "Confusion Matrix"
                from sklearn.metrics import plot_confusion_matrix
                disp = plot_confusion_matrix(classifier, X_test, y_test,
                                    display_labels=[0,1],
                                    cmap=plt.cm.Blues,
                                    normalize=None)
                disp.ax_.set_title(title)
                plt.savefig('static/plots/SLR/RF.png',orientation='landscape')
                return render(request, "design.html", {'rfsomething':True, 'wd':wd, 'cp':cp,'total':total,'score':score})
                

        else:
            return render(request, "design.html")

    else:
        corection = int(request.GET["no"])
        if corection == 1:
            iv = request.GET["inputvalue"]
            sm = joblib.load('regression_model')
            rm = int(sm.predict([[iv]]))
            return render(request, "design.html", {'someinner':True, 'si': rm,})
        elif corection == 2:
            iv = int(request.GET["inputvalue"])
            iv2 = int(request.GET["inputvalue2"])   
            sm = joblib.load('Classification_model')
            rm = int(sm.predict([[iv,iv2]]))
            return render(request, "design.html", {'lrsomething':True, 'si': rm})
        elif corection == 3:
            iv = int(request.GET["inputvalue"])
            iv2 = int(request.GET["inputvalue2"])   
            sm = joblib.load('Randomforest_model')
            rm = int(sm.predict([[iv,iv2]]))
            return render(request, "design.html", {'rfsomething2':True, 'si': rm})
        
        return render(request, 'design.html')

def upload(request):
    return render(request, 'fileupload.html')

def reset(request):
    return render(request,'fileupload.html')
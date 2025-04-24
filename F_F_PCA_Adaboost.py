from sklearn.decomposition   import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score
from sklearn                 import preprocessing
from sklearn.metrics         import plot_confusion_matrix
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import precision_score
from sklearn.metrics         import classification_report

import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np


#Load Data

df = pd.read_csv("C:/Users/ISD/Python_ali/fault_detection/jupyter/Pv_Data3.csv")
df;
X = df.drop(['Target','I1','I2','I3','I4','I5','I6','I7','I8'],axis=1).values
X1= df.drop(['Target','Voc','Isc','Vmp','Imp','Pmp','T','G'],axis=1).values
Y=df.drop(['Voc','Isc','Vmp','Imp','Pmp','T','G','I1','I2','I3','I4','I5','I6','I7','I8',],axis=1).values.ravel()
X.shape
Y=np.array([Y])
Y=Y.T
#preproccesing

pca = PCA(n_components=7)
pca.fit(X)
X_pca = pca.transform(X)

trainX, testX,trainY,testY = train_test_split(X_pca ,Y, test_size=0.2, random_state=0)

#creating model

model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=120,random_state=0)
model.fit(trainX,trainY)

##predict
#Train Data
ypred=model.predict(trainX)
ac1=accuracy_score(trainY,ypred)*100
pr1=precision_score(trainY, ypred, average=None)

#TestData
ypred1=model.predict(testX)
ac2=accuracy_score(testY,ypred1)*100
pr2=precision_score(testY, ypred1, average=None)

#All Data
YR=np.concatenate((trainY,testY),axis=0)
YP=np.concatenate((ypred,ypred1), axis=0)
Xo=np.concatenate((trainX,testX),axis=0)
ac3=accuracy_score(YR,YP)*100
pr3=precision_score(testY, ypred1, average=None)

cr=classification_report(YR,YP)

##plot Results 
#Train Data
fig = plot_confusion_matrix(model,trainX,trainY, display_labels=model.classes_)
fig.figure_.suptitle("Confusion Matrix for PV  Train Dataset")
plt.show()

#Test Data
fig = plot_confusion_matrix(model,testX,testY, display_labels=model.classes_)
fig.figure_.suptitle("Confusion Matrix for PV  Test Dataset")
plt.show()

#All Data
fig = plot_confusion_matrix(model,Xo,YR, display_labels=model.classes_)
fig.figure_.suptitle("Confusion Matrix for PV  All Dataset")
plt.show()
#c2=confusion_matrix(YR,YP)


#Location in PV Array   

#preproccesing
X1 = preprocessing.normalize(X1, norm='l2')                  
X1=np.concatenate((Y,X1),axis=1)
n=[]
r=[]

n1=[] 
r1=[]

n2=[]
r2=[]
t2=()

n3=[]
r3=[]

n4=[]
r4=[]

n5=[]
r5=[]

n6=[]
r6=[]

n7=[]
r7=[]

n8=[]
r8=[]

n9=[]
r9=[]

n10=[]
r10=[]
t10=()

for i in range(1600):
        if X1[i,0]==3:
            n3.append(i+2)
            for j in range(1,9):
                if(abs(X1[i,j]-sum(X1[i,1:])/8))>0.3:
                    r3.append(j)    
        elif X1[i,0]==4:
            n4.append(i+2)
            for j in range(1,9):
                if(abs(X1[i,j]-sum(X1[i,1:])/8))>0.3:
                    r4.append(j)
        elif X1[i,0]==5:
            n5.append(i+2)
            for j in range(1,9):
                if(abs(X1[i,j]-sum(X1[i,1:])/8))>0.003:
                    r5.append(j)
        elif X1[i,0]==6:
            n6.append(i+2)
            for j in range(1,9):
                if(abs(X1[i,j]-sum(X1[i,1:])/8))>0.1:
                    r6.append(j)
        elif X1[i,0]==7:
            n7.append(i+2)
            for j in range(1,9):
                if(abs(X1[i,j]-sum(X1[i,1:])/8))>0.2:
                    r7.append(j)
        elif X1[i,0]==8:
            n8.append(i+2)
            for j in range(1,9):
                if(abs(X1[i,j]-sum(X1[i,1:])/8))>0.7:
                    r8.append(j)
        elif X1[i,0]==9:
            n9.append(i+2)
            for j in range(1,9):
                if(abs(X1[i,j]-sum(X1[i,1:])/8))>0.005872:
                    r9.append(j)
        elif X1[i,0]==10:
                    n10.append(i+2)
                    for j in range(1,9):
                        if (abs(X1[i,j]-sum(X1[i,1:])/8))>0.306:
                            t10=t10+(j,)
                    r10.append(t10)
                    t10=()
        elif X1[i,0]==1:
            if X1[i,0]==1:
                n1.append(i+2)
                r1.append(0)
        elif X1[i,0]==2:
            n2.append(i+2)
            for j in range(1,9):
                if (abs(X1[i,j]-sum(X1[i,1:])/8))>0.11881:
                    t2=t2+(j,)
            r2.append(t2)
            t2=()
        
n.extend(n1)
r.extend(r1)
c1=1*np.ones(160)

n.extend(n2)
r.extend(r2)
c2=2*np.ones(160)
            
n.extend(n3)
r.extend(r3)
c3=3*np.ones(160)    

n.extend(n4)
r.extend(r4)
c4=4*np.ones(160)

n.extend(n5)
r.extend(r5)
c5=5*np.ones(160)

n.extend(n6)
r.extend(r6)
c6=6*np.ones(160)

n.extend(n7)
r.extend(r7)
c7=7*np.ones(160)

n.extend(n8)
r.extend(r8)
c8=8*np.ones(160)

n.extend(n9)
r.extend(r9)
c9=9*np.ones(160)

n.extend(n10)
r.extend(r10)
c10=10*np.ones(160)

c=np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10),axis=0)

r=np.array(r)
n=np.array(n)
out_put=[n,c,r]

output_dict={'number':n,'class':c,'string':r}
df_out=pd.DataFrame(output_dict)


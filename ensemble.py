from sklearn.decomposition   import PCA
from sklearn.model_selection import KFold
from sklearn.metrics         import accuracy_score
from sklearn                 import preprocessing
from sklearn.metrics         import plot_confusion_matrix
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import precision_score
from sklearn.metrics         import classification_report
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import VotingClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np


#Load Data
df = pd.read_csv("C:/Users/ISD/Python_ali/fault_detection/jupyter/Pv_Data3.csv")
df;
X = df.drop(['Target','I1','I2','I3','I4','I5','I6','I7','I8'],axis=1).values
X1= df.drop(['Target','Voc','Isc','Vmp','Imp','Pmp','T','G',],axis=1).values
Y=df.drop(['Voc','Isc','Vmp','Imp','Pmp','T','G','I1','I2','I3','I4','I5','I6','I7','I8',],axis=1).values.ravel()
X.shape


#preproccesing
pca = PCA(n_components=7)
pca.fit(X)
X= pca.transform(X)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3,random_state=42)

#model1 = DecisionTreeClassifier(criterion='gini',splitter='best', max_depth=11, random_state=0)
model2 = KNeighborsClassifier(n_neighbors=2)
model3 = SVC(kernel='linear' ,C=20, gamma='auto',probability=True)
model4 = RandomForestClassifier(n_estimators=41, max_depth=7, criterion='entropy' ,random_state=0)
model = VotingClassifier(estimators=[('knn',model2), ('svc',model3), ('RF',model4)],voting='soft', weights=[1, 2, 2])

#model1=model1.fit(trainX,trainY)
#model2=model2.fit(trainX,trainY)
#model3=model3.fit(trainX,trainY)
#model4=model4.fit(trainX,trainY)
model = model.fit(trainX,trainY)

ypred1=model.predict(trainX)
ac1=accuracy_score(trainY,ypred1)*100
pr1=precision_score(trainY, ypred1, average=None)

ypred2=model.predict(testX)
ac2=accuracy_score(testY,ypred2)*100
pr2=precision_score(testY, ypred2, average=None)

ypred3=model.predict(X)
ac3=accuracy_score(Y,ypred3)*100
pr3=precision_score(Y, ypred3, average=None)


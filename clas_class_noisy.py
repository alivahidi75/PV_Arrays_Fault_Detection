import random
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.decomposition   import PCA
from sklearn                 import datasets
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm             import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import accuracy_score
from sklearn.metrics         import plot_confusion_matrix
from sklearn.neural_network  import MLPClassifier
from sklearn                 import preprocessing
from sklearn.metrics         import plot_confusion_matrix
from sklearn.metrics         import precision_score
from sklearn.metrics         import classification_report


df = pd.read_csv("C:/Users/ISD/Python_ali/fault_detection/jupyter/Pv_Data3.csv")
df;
X = df.drop(['Target','I1','I2','I3','I4','I5','I6','I7','I8'],axis=1).values
X1= df.drop(['Target','Voc','Isc','Vmp','Imp','Pmp','T','G',],axis=1).values
Y=df.drop(['Voc','Isc','Vmp','Imp','Pmp','T','G','I1','I2','I3','I4','I5','I6','I7','I8',],axis=1).values
X.shape

Y_act=Y

# create noise
#target
l=[i for i in range(1600)]
random_unlabeled_points=random.sample(l,320)
l1=np.delete(l,random_unlabeled_points) 

Y_cler=np.delete(Y,random_unlabeled_points,0)
Y_n=np.delete(Y,l1,0)

#inputs
X_cler=np.delete(X,random_unlabeled_points,0)
X_n=np.delete(X,l1,0)

pca = PCA(n_components=7)
pca.fit(X_cler)
X_cler = pca.transform(X_cler)

trainX, testX,trainY,testY = train_test_split(X_cler ,Y_cler, test_size=0.3, random_state=0)

#model = RandomForestClassifier(n_estimators=120, max_depth=6, criterion='entropy' ,random_state=86)
model = DecisionTreeClassifier(criterion='entropy',splitter='best', max_depth=8, random_state=86)
model.fit(trainX,trainY)

##predict
#Train Data
ypred1=model.predict(trainX)
ac1=accuracy_score(trainY,ypred1)*100
pr1=precision_score(trainY, ypred1, average='macro')

#TestData
ypred2=model.predict(testX)
ac2=accuracy_score(testY,ypred2)*100
pr2=precision_score(testY, ypred2, average='macro')

#All Data
ypred3=model.predict(X_cler)
ac3=accuracy_score(Y_cler,ypred3)*100
pr3=precision_score(Y_cler,ypred3, average='macro')

cr=classification_report(Y_cler,ypred3)

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
fig = plot_confusion_matrix(model,X_cler,Y_cler, display_labels=model.classes_)
fig.figure_.suptitle("Confusion Matrix for PV  All Dataset")
plt.show()

#labeling the data
pca = PCA(n_components=7)
pca.fit(X_n)
X_n = pca.transform(X_n)
Y_pred=model.predict(X_n)

Y_pred=np.array([Y_pred])
Y_pred=Y_pred.T

Y_pre=np.concatenate((Y_pred,Y_cler), axis=0)
X_pre=np.concatenate((X_n,X_cler),axis=0)

#fault classification 

trainX_pre, testX_pre,trainY_pre,testY_pre = train_test_split(X_pre ,Y_pre, test_size=0.3, random_state=0)

model2 = RandomForestClassifier(n_estimators=120, max_depth=6, criterion='entropy' ,random_state=86)
#model2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=120,random_state=0)
model2.fit(trainX_pre,trainY_pre)

##predict
#Train Data
out1=model2.predict(trainX_pre)
ac_out1=accuracy_score(trainY_pre,out1)*100
pr_out1=precision_score(trainY_pre, out1, average='macro')

#TestData
out2=model2.predict(testX_pre)
ac_out2=accuracy_score(testY_pre,out2)*100
pr_out2=precision_score(testY_pre,out2, average='macro')

#All Data

out3=model2.predict(X_pre)
ac_out3=accuracy_score(Y_pre,out3)*100
pr_out3=precision_score(Y_pre,out3, average='macro')
cr_out=classification_report(Y_pre,out3)

##plot Results 
#Train Data
fig = plot_confusion_matrix(model,trainX_pre,trainY_pre, display_labels=model.classes_)
fig.figure_.suptitle("fault_detection:Confusion Matrix for PV  Train Dataset")
plt.show()

#Test Data
fig = plot_confusion_matrix(model,testX_pre,testY_pre, display_labels=model.classes_)
fig.figure_.suptitle("fault_detection:Confusion Matrix for PV  Test Dataset")
plt.show()

#All Data
fig = plot_confusion_matrix(model,X_pre,Y_pre, display_labels=model.classes_)
fig.figure_.suptitle("fault_detection:Confusion Matrix for PV  All Dataset")
plt.show()




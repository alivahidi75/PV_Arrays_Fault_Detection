from sklearn.decomposition   import PCA
from sklearn.model_selection import KFold
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
X1= df.drop(['Target','Voc','Isc','Vmp','Imp','Pmp','T','G',],axis=1).values
Y=df.drop(['Voc','Isc','Vmp','Imp','Pmp','T','G','I1','I2','I3','I4','I5','I6','I7','I8',],axis=1).values
X.shape


#preproccesing
pca = PCA(n_components=7)
pca.fit(X)
X= pca.transform(X)

k=5
kf=KFold(n_splits=k,random_state=None)
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=120,random_state=0)

acc_score_test=[]
acc_score_train=[]
acc_score_all=[]  

for train_ind,test_ind in kf.split(X):
    X_train, X_test = X[train_ind,:] , X[test_ind]
    Y_train, Y_test = Y[train_ind]   , Y[test_ind]
    
    model.fit(X_train,Y_train)
    
    pred_values_test=model.predict(X_test)
    pred_values_train=model.predict(X_train)
    pred_values_all=model.predict(X)
      
    acc_test=accuracy_score(pred_values_test,Y_test)
    acc_train=accuracy_score(pred_values_train,Y_train)
    acc_all=accuracy_score(pred_values_all,Y)
    
    acc_score_test.append(acc_test)
    acc_score_train.append(acc_train)
    acc_score_all.append(acc_all)
    
    
avrage_acc_score_test=np.mean(acc_score_test)    
avrage_acc_score_train=np.mean(acc_score_train)    
avrage_acc_score_all=np.mean(acc_score_all)

pred=model.predict(X)

acc=accuracy_score(pred,Y)


cr=classification_report(Y,pred)

fig = plot_confusion_matrix(model,X,Y, display_labels=model.classes_)
fig.figure_.suptitle("Confusion Matrix for PV  All Dataset")
plt.show()

t=[1,2,3,4,5]
plt.plot(t,acc_score_all)
plt.xlim(1,10)
plt.ylim(0.5,1)
plt.show()


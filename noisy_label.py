from numpy                   import concatenate
from sklearn.datasets        import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import plot_confusion_matrix
from sklearn.decomposition   import PCA
from sklearn.svm             import SVC

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

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


# split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)

# split train into labeled and unlabeled
X_train_lab, X_test_unlab, Y_train_lab, Y_test_unlab = train_test_split(X_train, Y_train, test_size=0.2, random_state=1, stratify=Y_train)

# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab))

# create "no label" for unlabeled data
nolabel =[[-1]for _ in range(len(Y_test_unlab))]
nolabel=np.array(nolabel)

# recombine training dataset labels
Y_train_mixed = concatenate((Y_train_lab, nolabel))

# define model
model = LabelPropagation()

# fit model on training dataset
model.fit(X_train_mixed, Y_train_mixed)

# get labels for entire training dataset data
tran_labels = model.transduction_

# define supervised learning model
#model2 = RandomForestClassifier(criterion='entropy',n_estimators=150,max_depth=8)
model2=RandomForestClassifier(n_estimators=41, max_depth=8, criterion='entropy' ,random_state=0)
#model2=SVC(kernel='linear' ,C=20, gamma='auto')
# fit supervised learning model on entire training dataset
model2.fit(X_train_mixed, tran_labels)

# make predictions on hold out test set
yhat_test = model2.predict(X_test)
yhat_train_mixed = model2.predict(X_train_mixed)
yhat_all = model2.predict(X)

# calculate score for test set
score_test = accuracy_score(Y_test, yhat_test)
score_train_mixed = accuracy_score(Y_train_mixed, yhat_train_mixed)
score_all = accuracy_score(Y, yhat_all)

# summarize score
print('Accuracy: %.3f' % (score_test*100))
print('Accuracy: %.3f' % (score_train_mixed*100))
print('Accuracy: %.3f' % (score_all*100))
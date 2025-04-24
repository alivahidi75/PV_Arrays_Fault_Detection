import numpy  as np
import pandas as pd
from sklearn.decomposition   import PCA
from sklearn                 import datasets
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm             import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score
from sklearn.metrics         import plot_confusion_matrix
from sklearn.neural_network  import MLPClassifier
import matplotlib.pyplot     as plt


df = pd.read_csv("C:/Users/ISD/Python_ali/fault_detection/jupyter/Pv_Data3.csv")
df;
X = df.drop(['Target','I1','I2','I3','I4','I5','I6','I7','I8'],axis=1).values
X1= df.drop(['Target','Voc','Isc','Vmp','Imp','Pmp','T','G',],axis=1).values
Y=df.drop(['Voc','Isc','Vmp','Imp','Pmp','T','G','I1','I2','I3','I4','I5','I6','I7','I8',],axis=1).values
X.shape

Y_act=Y

#target
l=[i for i in range(1600)]
random_unlabeled_points=np.random.choice(l,320)
Y[random_unlabeled_points]=-1

#inputs
pca = PCA(n_components=7)
pca.fit(X)
X= pca.transform(X)

trainX, testX,trainY,testY = train_test_split(X ,Y, test_size=0.2, random_state=56)

model=RandomForestClassifier(n_estimators=120, max_depth=7, criterion='entropy' ,random_state=86)
model.fit(trainX,trainY)

# random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
# iris.target[random_unlabeled_points] = -1
# target=iris.target

#model= SVC(probability=True, kernel='linear' ,C=20, gamma='auto')

#model=mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,60),max_iter = 300,activation = 'relu',solver = 'adam')

self_training_model = SelfTrainingClassifier(model, criterion='k_best', k_best=4)

self_training_model.fit(trainX,trainY)

#pred_train=self_training_model.predict(trainX)
#pred_test=self_training_model.predict(testX)
#pred_all=self_training_model.predict(X)

pred_all=model.predict(X)
ac_all=accuracy_score(Y_act,pred_all)

fig = plot_confusion_matrix(self_training_model,X,Y_act, display_labels=model.classes_)
fig.figure_.suptitle("Confusion Matrix for PV  All Dataset")
plt.show()










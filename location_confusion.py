import pandas as pd
import numpy as np
from sklearn                 import preprocessing

df = pd.read_csv("C:/Users/ISD/Python_ali/fault_detection/jupyter/exam.csv")
X1= df.drop(['T','L'],axis=1).values

Y=df.drop(['I1','I2','I3','I4','I5','I6','I7','I8','L'],axis=1).values.ravel()
Y=np.array([Y])
Y=Y.T

L=df.drop(['T','I1','I2','I3','I4','I5','I6','I7','I8'],axis=1).values.ravel()
L=np.array([L])
L=L.T

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


import pandas as pd
import numpy as np


df = pd.read_csv("C:/users/ISD/Python_ali/cods/spyder/data.csv")
arr = df.to_numpy()

a = []
for i in range(len(df)):
    for j in arr[i]:
        a.append(j[-3:])
        
b = np.array(a)


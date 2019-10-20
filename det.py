# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:56:54 2019

@author: mukmc
"""

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 

raw_data = loadarff('Aloi\\ALOI.arff')
df_data = pd.DataFrame(raw_data[0])

l=[]
for i in range(0,50000):
    
    if(raw_data[0][i][0].decode("utf-8")=='yes'):
        l.append(1)
    else:
        l.append(0)
    
df_data['outlier']=l

df_data.to_csv("raw.csv")
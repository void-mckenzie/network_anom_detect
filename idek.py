# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:53:35 2020

@author: mukmc
"""
import matplotlib.pyplot as plt

l1=dat_train["age"]**0.5
l2=dat_train["strength"]

l1, l2 = zip(*sorted(zip(l1, l2)))

plt.plot(l1,l2)

##############################################
import pandas as pd
import numpy as np

fak = pd.read_csv("daaat.csv")

dat_train=pd.read_csv("modified.csv")

x=fak.drop("strength",axis=1)
y=fak["strength"]

from sklearn.model_selection import train_test_split

xtrain,xval,ytrain,yval = train_test_split(x,y,test_size=0.3,random_state=75)

xtrain["cem/wat"]=xtrain["cement"]/xtrain["water"]

xval["cem/wat"]=xval["cement"]/xval["water"]

xtrain["age"]=xtrain["age"]
xval["age"]=xval["age"]

xtrain["na"]=xtrain["ca"]/xtrain["fa"]
xval["na"]=xval["ca"]/xval["fa"]




xtrain = xtrain[["blast","sp","age","flyash","cem/wat","na"]]
xval  = xval[["blast","sp","age","flyash","cem/wat","na"]]


df_test=pd.read_csv("test_data.csv")
df_test["cem/wat"]=df_test["cement"]/df_test["water"]
df_test["na"]=df_test["ca"]/df_test["fa"]

df_test= df_test[["blast","sp","age","flyash","cem/wat","na"]]
df_test = sc.transform(df_test)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xval = sc.transform(xval)


from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

cv = KFold(5, random_state = 1)

param_grid = {'n_estimators':[160,175,200,225,250,275,300],
              'max_depth':range(20,28,2), 
              'min_samples_split':range(125,135,1),
              'max_features':[5,6,7],
              'learning_rate':[0.2,0.199,0.195,0.19,0.185]}
clf = GridSearchCV(GradientBoostingRegressor(random_state=1), 
                   param_grid = param_grid, scoring='r2', 
                   cv=cv,n_jobs=-1,verbose=10).fit(xtrain, ytrain)
print(clf.best_estimator_) 
print("R Squared:",clf.best_score_)

gg=clf.best_estimator_

print("Test RMSE: ", np.sqrt(mean_squared_error(yval, gg.predict(xval))))

finpred = gboi.predict(df_test)


df_fin=pd.DataFrame()
df_fin['Id'] = range(0, len(df_test))
df_fin.set_index('Id')
df_fin['Predicted']=finpred
df_fin.to_csv("stacking_v2.csv")



from sklearn.ensemble import RandomForestRegressor
gboi = RandomForestRegressor(n_estimators=275,n_jobs=-1,max_depth=8,min_samples_split=2)
gboi.fit(x,y)

gboi.score(x,y)


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

model = CatBoostRegressor()
parameters = {'depth': [6,8,10],
                  'learning_rate' : [ 0.04, 0.06, 0.08],
                  'iterations'    : [625,600,650],
                  'l2_leaf_reg': [6, 8, 10,12]
                 }
grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 5, n_jobs=-1,verbose=10)
grid.fit(xtrain, ytrain)  
print(grid.best_estimator_) 
print("R Squared:",grid.best_score_)
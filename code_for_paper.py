# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:37:12 2020

@author: mukmc
"""
#Standard import cell

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras import regularizers
from keras.utils.data_utils import get_file
#%matplotlib inline




# Downloading training and test sets
try:
    training_set_path = get_file('KDDTrain%2B.csv', origin='https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv')
except:
    print('Error downloading')
    raise
    

try:
    test_set_path = get_file('KDDTest%2B.csv', origin='https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv')
except:
    print('Error downloading')
    raise
training_df = pd.read_csv(training_set_path, header=None)
testing_df = pd.read_csv(test_set_path, header=None)


training_df.head()

testing_df.head()

columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'difficulty'
]
training_df.columns = columns
testing_df.columns = columns

print("Training set has {} rows.".format(len(training_df)))
print("Testing set has {} rows.".format(len(testing_df)))

training_outcomes=training_df["outcome"].unique()
testing_outcomes=testing_df["outcome"].unique()
print("The training set has {} possible outcomes \n".format(len(training_outcomes)) )
print(", ".join(training_outcomes)+".")
print("\nThe testing set has {} possible outcomes \n".format(len(testing_outcomes)))
print(", ".join(testing_outcomes)+".")

# A list ot attack names that belong to each general attack type
dos_attacks=["snmpgetattack","back","land","neptune","smurf","teardrop","pod","apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpguess","worm","httptunnel","named","xlock","xsnoop","sendmail","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit","xterm","ps"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]

# Our new labels
classes=["Normal","Dos","R2L","U2R","Probe"]

#Helper function to label samples to 5 classes
def label_attack (row):
    if row["outcome"] in dos_attacks:
        return classes[1]
    if row["outcome"] in r2l_attacks:
        return classes[2]
    if row["outcome"] in u2r_attacks:
        return classes[3]
    if row["outcome"] in probe_attacks:
        return classes[4]
    return classes[0]


#We combine the datasets temporarily to do the labeling 
test_samples_length = len(testing_df)
df=pd.concat([training_df,testing_df])
df["Class"]=df.apply(label_attack,axis=1)


# The old outcome field is dropped since it was replaced with the Class field, the difficulty field will be dropped as well.
df=df.drop("outcome",axis=1)
df=df.drop("difficulty",axis=1)

# we again split the data into training and test sets.
training_df= df.iloc[:-test_samples_length, :]
testing_df= df.iloc[-test_samples_length:,:]


training_outcomes=training_df["Class"].unique()
testing_outcomes=testing_df["Class"].unique()
print("The training set has {} possible outcomes \n".format(len(training_outcomes)) )
print(", ".join(training_outcomes)+".")
print("\nThe testing set has {} possible outcomes \n".format(len(testing_outcomes)))
print(", ".join(testing_outcomes)+".")


# Helper function for scaling continous values
def minmax_scale_values(training_df,testing_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit([np.array(training_df[col_name])])
    train_values_standardized = scaler.transform([np.array(training_df[col_name])])
    training_df[col_name] = train_values_standardized[0]
    test_values_standardized = scaler.transform([np.array(testing_df[col_name])])
    testing_df[col_name] = test_values_standardized[0]
    

#Preprocessing
    
sympolic_columns=["protocol_type","service","flag"]
label_column="Class"
        

ntr=training_df[sympolic_columns]
ytr=training_df["Class"]
training_df=training_df.drop("Class",axis=1)
training_df=training_df.drop(sympolic_columns,axis=1)
nte = testing_df[sympolic_columns]
ytest=testing_df["Class"]
testing_df = testing_df.drop("Class",axis=1)
testing_df=testing_df.drop(sympolic_columns,axis=1)

lol = training_df.columns
scaler = MinMaxScaler()
scaler=scaler.fit(training_df)
training_df = scaler.transform(training_df)
training_df = pd.DataFrame(training_df,columns=lol)
testing_df=scaler.transform(testing_df)
testing_df = pd.DataFrame(testing_df,columns=lol)


temptrain = pd.get_dummies(ntr)
temptest=pd.get_dummies(nte)
l=[]
for i in temptrain.columns:
    if i not in temptest.columns:
        l.append(i)

for i in l:
    temptest[i]=np.zeros(temptest.shape[0])

temptrain = temptrain[temptrain.columns]
temptest = temptest[temptrain.columns]

training_df = pd.concat([training_df,temptrain],axis=1)
testing_df = pd.concat([testing_df,temptest],axis=1)
training_df["Class"]=ytr
testing_df["Class"]=ytest
training_df.head(5)

testing_df.head(5)

x,y=training_df,training_df.pop("Class").values
x=x.values
x_test,y_test=testing_df,testing_df.pop("Class").values
x_test=x_test.values
y0=np.ones(len(y),np.int8)
y0[np.where(y==classes[0])]=0
y0_test=np.ones(len(y_test),np.int8)
y0_test[np.where(y_test==classes[0])]=0

x.shape
x_test.shape
y.shape
y_test.shape


#Buildling and training the Denoising autoencoder model

def getModel():
    inp = Input(shape=(x.shape[1],))
    d1=Dropout(0.3)(inp)
    d2 = Dense(64, activation = 'relu', activity_regularizer = regularizers.l2(10e-5))(d1)
    encoded = Dense(8, activation='relu', activity_regularizer=regularizers.l2(10e-5))(d2)
    d3=Dense(64,activation="relu")(encoded)
    decoded = Dense(x.shape[1], activation='relu')(d3)
    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


denoise_autoencoder=getModel()
denoise_autoencoder.summary()

history=denoise_autoencoder.fit(x[np.where(y0==0)],x[np.where(y0==0)],
               epochs=5,
                batch_size=100,
                shuffle=True,
                validation_split=0.1
                       )

# Helper function that calculates the reconstruction loss of each data sample
def calculate_losses(x,preds):
    losses=np.zeros(len(x))
    for i in range(len(x)):
        losses[i]=((preds[i] - x[i]) ** 2).mean(axis=None)
        
    return losses

# We set the threshold equal to the training loss of the autoencoder
threshold=history.history["loss"][-1]

testing_set_predictions=denoise_autoencoder.predict(x_test)
test_losses=calculate_losses(x_test,testing_set_predictions)
testing_set_predictions=np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses>threshold)]=1


accuracy=accuracy_score(y0_test,testing_set_predictions)
recall=recall_score(y0_test,testing_set_predictions)
precision=precision_score(y0_test,testing_set_predictions)
f1=f1_score(y0_test,testing_set_predictions)
print("Performance over the testing data set \n")
print("Accuracy : {} , Recall : {} , Precision : {} , F1 : {}\n".format(accuracy,recall,precision,f1 ))



for class_ in classes:
    print(class_+" Detection Rate : {}".format(len(np.where(np.logical_and(testing_set_predictions==1 , y_test==class_))[0])/len(np.where(y_test==class_)[0])))


#Plotting confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

c = confusion_matrix(y0_test,testing_set_predictions)
plot_confusion_matrix(c,["Normal","Attack"])


plt.ylabel('Loss')
plt.xticks(np.arange(0,5), classes)
plt.violinplot([test_losses[np.where(y_test==class_)] for class_ in classes],np.arange(0,len(classes)),showmeans =True )
plt.axhline(y=threshold,c='r',label="Threshold Value")
plt.legend()


#Training the Deep Autoencoder model


autoencoder = Sequential()
autoencoder.add(Dense(64, input_shape=(x.shape[1],), activation = "relu",activity_regularizer=regularizers.l2(10e-3) ))
autoencoder.add(Dense(8, activation="relu"))
autoencoder.add(Dense(64, input_shape=(x.shape[1],), activation = "relu"))
aytoencoder.add(Dense(x.shape[1], activation='relu'))

autoencoder.compile(optimizer="nadam", loss="mean_squared_error")

autoencoder.fit(x[np.where(y0==0)],x[np.where(y0==0)],
               epochs=14,
                batch_size=32,
                shuffle=True,
                validation_split=0.1
                       )

threshold=0.0035

testing_set_predictions=autoencoder.predict(x_test)
test_losses=calculate_losses(x_test,testing_set_predictions)
testing_set_predictions=np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses>threshold)]=1

accuracy=accuracy_score(y0_test,testing_set_predictions)
recall=recall_score(y0_test,testing_set_predictions)
precision=precision_score(y0_test,testing_set_predictions)
f1=f1_score(y0_test,testing_set_predictions)
print("Performance over the testing data set \n")
print("Accuracy : {} , Recall : {} , Precision : {} , F1 : {}\n".format(accuracy,recall,precision,f1 ))

for class_ in classes:
    print(class_+" Detection Rate : {}".format(len(np.where(np.logical_and(testing_set_predictions==1 , y_test==class_))[0])/len(np.where(y_test==class_)[0])))

plt.ylabel('Loss')
plt.xticks(np.arange(0,5), classes)
plt.violinplot([test_losses[np.where(y_test==class_)] for class_ in classes],np.arange(0,len(classes)),showmeans =True )
plt.axhline(y=threshold,c='r',label="Threshold Value")
plt.legend()




#Training the Sparse Deep Autoencoder model

sae = Sequential()
sae.add(Dense(150, input_shape=(x.shape[1],), activation = "relu",activity_regularizer=regularizers.l2(10e-3) ))
sae.add(Dense(122, activation="relu"))

sae.compile(optimizer="nadam", loss="mean_squared_error")

sae.fit(x[np.where(y0==0)],x[np.where(y0==0)],
               epochs=14,
                batch_size=32,
                shuffle=True,
                validation_split=0.1
                       )

threshold=0.0035

testing_set_predictions=sae.predict(x_test)
test_losses=calculate_losses(x_test,testing_set_predictions)
testing_set_predictions=np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses>threshold)]=1

accuracy=accuracy_score(y0_test,testing_set_predictions)
recall=recall_score(y0_test,testing_set_predictions)
precision=precision_score(y0_test,testing_set_predictions)
f1=f1_score(y0_test,testing_set_predictions)
print("Performance over the testing data set \n")
print("Accuracy : {} , Recall : {} , Precision : {} , F1 : {}\n".format(accuracy,recall,precision,f1 ))

for class_ in classes:
    print(class_+" Detection Rate : {}".format(len(np.where(np.logical_and(testing_set_predictions==1 , y_test==class_))[0])/len(np.where(y_test==class_)[0])))

plt.ylabel('Loss')
plt.xticks(np.arange(0,5), classes)
plt.violinplot([test_losses[np.where(y_test==class_)] for class_ in classes],np.arange(0,len(classes)),showmeans =True )
plt.axhline(y=threshold,c='r',label="Threshold Value")
plt.legend()



#Training the Sparse Deep Denoising Autoencoder model

inp = Input(shape=(x.shape[1],))
d1=Dropout(0.3)(inp)
d2 = Dense(144,activation='relu', activity_regularizer= regularizers.l2(10e-4))(d1)
d3 = Dense(150, activation = 'relu', activity_regularizer = regularizers.l2(10e-4))(d2)
d4 = Dense(122, activation='relu')(d3)

sdae=Model(inp,d4)

sdae.compile(optimizer="nadam", loss="mean_squared_error")

sdae.fit(x[np.where(y0==0)],x[np.where(y0==0)],
               epochs=14,
                batch_size=32,
                shuffle=True,
                validation_split=0.1
                       )
threshold=0.0035

testing_set_predictions=sdae.predict(x_test)
test_losses=calculate_losses(x_test,testing_set_predictions)
testing_set_predictions=np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses>threshold)]=1

accuracy=accuracy_score(y0_test,testing_set_predictions)
recall=recall_score(y0_test,testing_set_predictions)
precision=precision_score(y0_test,testing_set_predictions)
f1=f1_score(y0_test,testing_set_predictions)
print("Performance over the testing data set \n")
print("Accuracy : {} , Recall : {} , Precision : {} , F1 : {}\n".format(accuracy,recall,precision,f1 ))

for class_ in classes:
    print(class_+" Detection Rate : {}".format(len(np.where(np.logical_and(testing_set_predictions==1 , y_test==class_))[0])/len(np.where(y_test==class_)[0])))

plt.ylabel('Loss')
plt.xticks(np.arange(0,5), classes)
plt.violinplot([test_losses[np.where(y_test==class_)] for class_ in classes],np.arange(0,len(classes)),showmeans =True )
plt.axhline(y=threshold,c='r',label="Threshold Value")
plt.legend()


 
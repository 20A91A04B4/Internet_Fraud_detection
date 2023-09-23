# Internet_Fraud_detection
#STEP1:DATA READING

import pandas as pd
import numpy as np

data=pd.read_csv(r"D:\projectspace4.0\onlinefraud.csv")


##STEP2: DATA UNDERSTANDING

data.shape       #no o rows and columns
data.size        #total no of cells
data.info()      #rows,columns,null spaces...all information
data.describe()  #statistical values
data.head()      #top 5 values   
data.dropna()    #delete empty rows

                

##STEP3:DATA PREPROSESSING

##sum of empty in columns
data.isna().sum()
#deleting columns with receipts or unnecessary
data.drop(['nameOrig'],axis=1,inplace=True)
data.drop(['nameDest'],axis=1,inplace=True)
data.drop(['isFlaggedFraud'],axis=1,inplace=True)
data.dtypes
##data contains strings to convert the strings into integers use LabelEncoder
data['type'].value_counts()
data["type"]=data["type"].map({"CASH_OUT":1,"PAYMENT":2,"CASH_IN":3,"TRANSFER":4,"DEBIT":5})

## No of count of type and whether the transction is fraud or not

data['type'].value_counts()
data['isFraud'].value_counts()
data.columns

##STEP4:MODEL BUILDING

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

##Dividing data into training and testing

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9)
#20% out of all
#1st running accuracy will come every time by using random_state=any number

## Reshaping the data for convolution
xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
xtest=xtest.reshape(xtest.shape[0],xtest.shape[1],1)
xtrain.shape,xtest.shape


#############                    CNN ALGORITHM                  ##############

import tensorflow as tf
from tensorflow import keras     #keras is an interface for tensorflow
from tensorflow.keras import Sequential   #sequential means neural network
from tensorflow.keras.layers import Flatten,Dense

model=Sequential()

# NEURAL NETWORKS----->1.input layer    2.hidden layer     3.output layer
model.add(Dense(xtrain.shape[1],activation='relu',input_dim=xtrain.shape[1]))  
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


##STEP5: MODEL EVAULATION

## to minimize error we use optimizer=adam
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=3)

ypred=model.predict(xtest)


##Serilize model to json model
model_json = model.to_json()
with open("modern.json", "w") as json_file:
    json_file.write(model_json)
    
#serialize weights to HDF5
model.save_weights("modern.h5")
print("Saved model to disk")


##TESTING

import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from time import sleep

from keras.models import model_from_json
import tensorflow as tf


json_file=open(r"C:\Users\91630\modern.json",'r')
loaded_model_json=json_file.read()
json_file.close()

#to read json file
from tensorflow.keras.models import model_from_json
loaded_model=model_from_json(loaded_model_json)

#load weights into new model
#to read h5 file
loaded_model.load_weights(r"C:\Users\91630\modern.h5")
print('loaded_model from disk')

if (loaded_model.predict([[1,3,1557555346.00,5638.00,4035679.00,46000.00,32456765.00]])):
    print("FRAUD")
else:
    print("NOT A FRAUD")




## Detecting Fraud Transaction




####################                KNN ALGORITHM                   ##################

#system reading the data
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)

#predection

ypred=model.predict(xtest)

# finding accuracy by comparing ytest(original) and ypred
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

# Checking output for predection values
print(loaded_model.predict([[3,2,8900.2,8990.2,3.4,46688.0,51524.66713]]))


######################             SVM  ALGORITHM                  ####################### 

from sklearn.svm import SVC
model=SVC(kernel='linear')   ###linear,poly,rbf


model.fit(xtrain,ytrain)
ypred=model.predict(xtest) #prediction values

#comparing prediction values given by algorithm and testing values and checking for accuracy

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100) 

#Giving output for predection values

print(model.predict([[1,2,5366.1,84557.4447777,569971.25,567789.21,22254.0]]))

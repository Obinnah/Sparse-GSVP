from scipy.io import arff
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def get_train_test_val(X,y,rho,eta):
    Xtrain,Xv,ytrain,yv = train_test_split(X,y,test_size=rho, random_state=42) # rho=0.3
    Xval,Xtest,yval,ytest = train_test_split(Xv,yv,test_size=eta, random_state=42) # eta=0.1
    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def generator1(datavar):
    if datavar == 'breast_cancer':
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/breast-cancer.csv')
        label = data['diagnosis'].replace({'B':0,'M':1})
        AA = data.iloc[0:len(label[label==0]),2:data.shape[1]].values
        BB = data.iloc[len(label[label==0]):data.shape[0],2:data.shape[1]].values
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.4)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.4)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = data.columns[2:].to_list()
        
    elif datavar == 'Ovarian':
        data, meta = arff.loadarff('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Ovarian.arff')
        Datta = pd.DataFrame(data) # the last column is the class
        label = Datta['Class'].replace({b'Normal':0,b'Cancer':1})
        Data = Datta.iloc[label.argsort()[::1],:] # Sort the rows of the data
        AA = Data.iloc[0:len(label[label==0]),0:Datta.shape[1]-1].values # Normal class
        BB = Data.iloc[len(label[label==0]):Data.shape[0],0:Datta.shape[1]-1].values # Cancer class
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.3)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.3)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = Datta.columns[:-1].to_list()

    os.chdir('C:/Users/uugob/GSVP/Training_data') 
    np.save('X_train_{}.npy'.format(os.path.splitext(datavar)[0]),Xtrain)
    np.save('y_train_{}.npy'.format(os.path.splitext(datavar)[0]),ytrain)
    os.chdir('C:/Users/uugob/GSVP/Validation_data')
    np.save('X_val_{}.npy'.format(os.path.splitext(datavar)[0]),Xval)
    np.save('y_val_{}.npy'.format(os.path.splitext(datavar)[0]),yval)
    os.chdir('C:/Users/uugob/GSVP/Testing_data')
    np.save('X_test_{}.npy'.format(os.path.splitext(datavar)[0]), Xtest)
    np.save('y_test_{}.npy'.format(os.path.splitext(datavar)[0]), ytest)
    os.chdir('C:/Users/uugob/GSVP/dataheads')
    if datavar in ['breast_cancer',  'Ovarian']:
        np.save('heading_{}.npy'.format(os.path.splitext(datavar)[0]),headings)


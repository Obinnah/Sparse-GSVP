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


def generator0(datavar):
    if datavar == 'Camel':
        AA = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Camel_BLD.csv').iloc[:,1:40].values.T
        BB = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Camel_NAS.csv').iloc[:,1:40].values.T
        GM = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Alpaca_genes.csv').iloc[:,1:49].values.T
        qa = 9      
        qb = 12
        XA = AA[0:qa,:] # controls
        XB = AA[qa:qa+qb,:] # infected
        Xvalo,Xtesto,yvalo,ytesto = train_test_split(XA,[0]*XA.shape[0],test_size=0.1, random_state=42)
        Xvale,Xteste,yvale,yteste = train_test_split(XB,[1]*XB.shape[0],test_size=0.1, random_state=42)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        Xtrain = BB[0:qa+qb,:]
        ytrain = np.concatenate((np.array([0]*qa),np.array([1]*qb)))
        headings = GM.flatten().tolist()
        print(headings)


    elif datavar == 'Alpaca':
        AA = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Alpaca_BLD.csv').iloc[:,1:49].values.T
        BB = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Alpaca_NAS.csv').iloc[:,1:49].values.T
        GM = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Alpaca_genes.csv').iloc[:,1:49].values.T
        qa = 12     
        qb = 16

        XA = AA[0:qa,:] # controls
        XB = AA[qa:qa+qb,:] # infected
        Xvalo,Xtesto,yvalo,ytesto = train_test_split(XA,[0]*XA.shape[0],test_size=0.1, random_state=42)
        Xvale,Xteste,yvale,yteste = train_test_split(XB,[1]*XB.shape[0],test_size=0.1, random_state=42)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        Xtrain = BB[0:qa+qb,:]
        ytrain = np.concatenate((np.array([0]*qa),np.array([1]*qb)))
        headings = GM.flatten().tolist()

    elif datavar == 'Camelid':
        AA = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/BLD_Merged.csv').iloc[:,1:50].values.T
        BB = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/NAS_Merged.csv').iloc[:,1:50].values.T
        GM = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Meta_Merged.csv').iloc[:,1:49].values.T
        qa = 21      
        qb = 28

        XA = AA[0:qa,:] # controls
        XB = AA[qa:qa+qb,:] # infected
        Xvalo,Xtesto,yvalo,ytesto = train_test_split(XA,[0]*XA.shape[0],test_size=0.1, random_state=42)
        Xvale,Xteste,yvale,yteste = train_test_split(XB,[1]*XB.shape[0],test_size=0.1, random_state=42)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        Xtrain = BB[0:qa+qb,:]
        ytrain = np.concatenate((np.array([0]*qa),np.array([1]*qb)))
        headings = GM.flatten().tolist()

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
    if datavar in ['Camel', 'Alpaca',  'Camelid']:
        np.save('heading_{}.npy'.format(os.path.splitext(datavar)[0]),headings)


def generator1(datavar):
    if datavar == 'diabetes':
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/diabetes.csv')
        label = data['Outcome'] # condition
        AA = data.iloc[0:len(label[label==0]),0:data.shape[1]-1].values
        BB = data.iloc[len(label[label==0]):data.shape[0],0:data.shape[1]-1].values
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = data.columns[:-1].to_list()
        
    elif datavar == 'heart_cleveland':
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/heart_cleveland.csv')
        label = data['condition'] # condition
        AA = data.iloc[0:len(label[label==0]),0:data.shape[1]-1].values
        BB = data.iloc[len(label[label==0]):data.shape[0],0:data.shape[1]-1].values
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = data.columns[:-1].to_list()
        
    elif datavar == 'breast_cancer':
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/breast-cancer.csv')
        label = data['diagnosis'].replace({'B':0,'M':1})
        AA = data.iloc[0:len(label[label==0]),2:data.shape[1]].values
        BB = data.iloc[len(label[label==0]):data.shape[0],2:data.shape[1]].values
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
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
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = Datta.columns[:-1].to_list()
               
    elif datavar == 'Prostate_GE':
        Kingry = scipy.io.loadmat('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Prostate_GE.mat') 
        Data = Kingry['X']
        label = Kingry['Y'] # This is ordered already
        AA = Data[0:len(label[label==1]),:] # Normal class
        BB = Data[len(label[label==1]):Data.shape[0],:] # Cancer class
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
               
    elif datavar == 'SMK_CAN_187':
        Kingry = scipy.io.loadmat('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/SMK_CAN_187.mat')
        Data = Kingry['X']
        label = Kingry['Y'] # This is ordered already
        AA = Data[0:len(label[label==1]),:] # Without cancer
        BB = Data[len(label[label==1]):Data.shape[0],:] # With cancer
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        
    elif datavar == 'GLI_85':
        Kingry = scipy.io.loadmat('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/GLI_85.mat') 
        data = Kingry['X'] 
        label = Kingry['Y'].flatten() # This is ordered already
        AA = data[0:len(label[label==1]),:] # Normal class
        BB = data[len(label[label==1]):data.shape[0],:] # tumor class
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        
    elif datavar == 'Leukemia':
        data, meta = arff.loadarff('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/Leukemia.arff')
        Datta = pd.DataFrame(data)
        label = Datta['CLASS'].replace({b'ALL':0,b'AML':1})
        Data = Datta.iloc[label.argsort()[::1],:]
        AA = Data.iloc[0:len(label[label==0]),0:Datta.shape[1]-1].values # Normal class
        BB = Data.iloc[len(label[label==0]):Data.shape[0],0:Datta.shape[1]-1].values # tumor class
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = Datta.columns[:-1].to_list()
        
    elif datavar == 'Influenza_I':
        Kingry = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_data.csv') # data matrix
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_metadata.csv') # This is the meta data
        d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
        d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
        label1 = data[0:863] # meta data associated with h1n1
        label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
        nu = 0
        za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
        zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
        dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
        dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
        dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
        dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
        AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
        BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(AA,[0]*AA.shape[0],0.25,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(BB,[1]*BB.shape[0],0.25,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = Kingry.columns[1:].to_list()
        
    elif datavar == 'Influenza_II':
        Kingry = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_data.csv') # data matrix
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_metadata.csv') # This is the meta data
        d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
        d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
        label1 = data[0:863] # meta data associated with h1n1
        label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
        nu = 0
        za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
        zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
        dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
        dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
        dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
        dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
        AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
        BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
        qa = len(za[za==0]) # getting number of non-shedders in h1n1
        qb = len(zb[zb==0]) # getting number of non-shedders in h3n2
        Xtrain = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data = all non-shedders
        ytrain = np.concatenate(([0]*qa,[1]*qb))
        XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data = all shedders
        Xval, Xtest,yval,ytest = train_test_split(XZ,np.concatenate(([0]*(AA.shape[0]-qa),[1]*(BB.shape[0]-qb)),axis=0),test_size=0.1, random_state=42)
        headings = Kingry.columns[1:].to_list() 

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
    if datavar in ['diabetes', 'heart_cleveland',  'breast_cancer',  'Ovarian','Leukemia', 'Influenza_I', 'Influenza_II']:
        np.save('heading_{}.npy'.format(os.path.splitext(datavar)[0]),headings)

def generator2(datavar):
    if datavar == 'Influenza_III':
        Kingry = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_data.csv') # data matrix
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_metadata.csv') # This is the meta data
        d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
        d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
        label1 = data[0:863] # meta data associated with h1n1
        label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
        nu = 0
        za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
        zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
        dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
        dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
        dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
        dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
        AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
        BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
        qa = len(za[za==0]) # getting number of non-shedders in h1n1
        qb = len(zb[zb==0]) # getting number of non-shedders in h3n2
        Xtrain = np.concatenate((AA[0:qa,:],AA[qa:AA.shape[0],:]),axis=0) # training data = h1n1 non-shedders/shedders
        ytrain = np.concatenate(([0]*qa,[1]*(AA.shape[0]-qa)))
        XZ = np.concatenate((BB[0:qb,:],BB[qb:BB.shape[0],:]),axis=0) # testing data = h3n2 non-shedders/shedders
        Xval, Xtest,yval,ytest = train_test_split(XZ,np.concatenate(([0]*qb,[1]*(BB.shape[0]-qb)),axis=0),test_size=0.1, random_state=42)
        headings = Kingry.columns[1:].to_list()
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
    np.save('heading_{}.npy'.format(os.path.splitext(datavar)[0]),headings)

def generator3(datavar):
    if datavar == 'Influenza_IV':
        Kingry = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_data.csv') # data matrix
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_metadata.csv') # This is the meta data
        d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
        d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
        label1 = data[0:863] # meta data associated with h1n1
        label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
        nu = 0
        za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
        zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
        dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
        dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
        dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
        dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
        AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
        BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
        xa = len(za[za==0]) # getting number of non-shedders in h1n1
        xb = len(zb[zb==0]) # getting number of non-shedders in h3n2
        XA = np.concatenate((AA[0:xa,:],BB[0:xb,:]),axis=0) # combining h1n1 and h3n2 non-shedder data
        XB = np.concatenate((AA[xa:AA.shape[0],:],BB[xb:BB.shape[0],:]),axis=0) # combining h1n1 and h3n2 shedder data
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(XA,[0]*XA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(XB,[1]*XB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = Kingry.columns[1:].to_list()
        
    elif datavar == 'Influenza_V':
        Kingry = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_data.csv') # data matrix
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_metadata.csv') # This is the meta data
        d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
        d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
        label1 = data[0:863] # meta data associated with h1n1
        label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
        nu = 0
        nnu = 24 # upperbound for time_id
        za = label1[(label1['time_id']>0)&(label1['time_id']<=nnu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0) and bounded, then convert T/F to 1/0
        zb = label2[(label2['time_id']>0)&(label2['time_id']<=nnu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0) and bounded, then convert T/F to 1/0
        dd1 = d1.iloc[label1[(label1['time_id']>0)&(label1['time_id']<=nnu)].index,:] # Use the indices for 0<time_id<=24 to select rows in h1n1
        dd2 = d2.iloc[label2[(label2['time_id']>0)&(label2['time_id']<=nnu)].index,:] # Use the indices for 0<time_id<=24 to select rows in h3n2
        dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
        dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
        AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
        BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
        xa = len(za[za==0]) # getting number of non-shedders in h1n1
        xb = len(zb[zb==0]) # getting number of non-shedders in h3n2
        XA = np.concatenate((AA[0:xa,:],BB[0:xb,:]),axis=0) # combining h1n1 and h3n2 non-shedder data
        XB = np.concatenate((AA[xa:AA.shape[0],:],BB[xb:BB.shape[0],:]),axis=0) # combining h1n1 and h3n2 shedder data
        Xtraino, ytraino, Xvalo, yvalo, Xtesto, ytesto = get_train_test_val(XA,[0]*XA.shape[0],0.3,0.1)
        Xtraine, ytraine, Xvale, yvale, Xteste, yteste = get_train_test_val(XB,[1]*XB.shape[0],0.3,0.1)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        Xtest = np.concatenate((Xtesto,Xteste),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        ytest = np.concatenate((ytesto,yteste),axis=0)
        headings = Kingry.columns[1:].to_list()
        
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
    if datavar in [ 'Influenza_IV', 'Influenza_V']:
        np.save('heading_{}.npy'.format(os.path.splitext(datavar)[0]),headings)

def generator4(datavar):
    if datavar == 'Influenza_VI':
        Kingry = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_data.csv') # data matrix
        data = pd.read_csv('H:/My Drive/KIRBY PAPERS/GSVD final Analysis/Tularensis/Tularemia/MERS-COV/gse73072_metadata.csv') # This is the meta data
        d1 = Kingry.iloc[477:863,1:Kingry.shape[1]].reset_index(drop=True) # Note h1n1 dee3 data [0:477] excluded for testing
        d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
        label1 = data[477:863].reset_index(drop=True) # meta data associated with h1n1
        label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
        nu = 0
        nnu = 24 # upperbound for time_id
        za = label1[(label1['time_id']>0)&(label1['time_id']<=nnu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), bounded, then convert T/F to 1/0
        zb = label2[(label2['time_id']>0)&(label2['time_id']<=nnu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), bounded, then convert T/F to 1/0
        dd1 = d1.iloc[label1[(label1['time_id']>0)&(label1['time_id']<=nnu)].index,:] #Use the indices for 0<time_id<=24 to select rows in h1n1
        dd2 = d2.iloc[label2[(label2['time_id']>0)&(label2['time_id']<=nnu)].index,:] # Use the indices for 0<time_id<=0 to select rows in h3n2
        dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
        dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
        AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
        BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
        xa = len(za[za==0]) # getting number of non-shedders in h1n1
        xb = len(zb[zb==0]) # getting number of non-shedders in h3n2
        XA = np.concatenate((AA[0:xa,:],BB[0:xb,:]),axis=0) # combining h1n1 and h3n2 non-shedder data
        XB = np.concatenate((AA[xa:AA.shape[0],:],BB[xb:BB.shape[0],:]),axis=0) # combining h1n1 and h3n2 shedder data
        Xtraino,Xvalo,ytraino,yvalo = train_test_split(XA,[0]*XA.shape[0],test_size=0.3, random_state=42)
        Xtraine,Xvale,ytraine,yvale = train_test_split(XB,[0]*XB.shape[0],test_size=0.3, random_state=42)
        Xtrain = np.concatenate((Xtraino,Xtraine),axis=0)
        Xval = np.concatenate((Xvalo,Xvale),axis=0)
        ytrain = np.concatenate((ytraino,ytraine),axis=0)
        yval = np.concatenate((yvalo,yvale),axis=0)
        headings = Kingry.columns[1:].to_list()
        
        c1 = Kingry.iloc[0:477,1:Kingry.shape[1]].reset_index(drop=True) # Note h1n1 dee3 data [0:477] excluded for testing
        label1 = data[0:477].reset_index(drop=True) # meta data associated with test h1n1 dee3
        zc = label1[(label1['time_id']>0)&(label1['time_id']<=nnu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), bounded, then convert T/F to 1/0
        cc1 = c1.iloc[label1[(label1['time_id']>0)&(label1['time_id']<=nnu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
        can1 = np.argsort(zc) # get indices for h1n1 shedders(1)/non-shedders(0)
        XZ = cc1.iloc[can1,:].values # Sorting h1n1 with non-shedders up/shedders down
        cb = len(zc[zc==0]) # getting number of non-shedders in h1n1

    os.chdir('C:/Users/uugob/GSVP/Training_data') 
    np.save('X_train_{}.npy'.format(os.path.splitext(datavar)[0]),Xtrain)
    np.save('y_train_{}.npy'.format(os.path.splitext(datavar)[0]),ytrain)
    os.chdir('C:/Users/uugob/GSVP/Validation_data')
    np.save('X_val_{}.npy'.format(os.path.splitext(datavar)[0]),Xval)
    np.save('y_val_{}.npy'.format(os.path.splitext(datavar)[0]),yval)
    os.chdir('C:/Users/uugob/GSVP/Testing_data')
    np.save('X_test_{}.npy'.format(os.path.splitext(datavar)[0]),XZ)
    np.save('y_test_{}.npy'.format(os.path.splitext(datavar)[0]),np.concatenate((np.array([0]*cb),np.array([1]*(XZ.shape[0]-cb)))))
    os.chdir('C:/Users/uugob/GSVP/dataheads')
    np.save('heading_{}.npy'.format(os.path.splitext(datavar)[0]),headings)



 
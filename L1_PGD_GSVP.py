# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:30:00 2023

@author: ugoob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
import random
from sklearn import svm
from kneefinder import KneeFinder
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


def threshold(x,epsa,q):
    return (x**2 + epsa**2)**((q-2)/2)

def proximal_gradient_descent(grad_f, prox_h, g,A,B,x_initial, step_size, lambd,num_iterations):
    x0 = x_initial
    for _ in range(num_iterations):
        gradient = grad_f(A,B,x0,g)
        x = prox_h(x0 - step_size * gradient, step_size,lambd)
        if np.linalg.norm(x-x0)/np.linalg.norm(x0)<1e-4:
            print('condition satisfied')
            break 
        else:
            x0 = x
    return x

def g(A,B,x):
    return np.linalg.norm(A@x)**2/np.linalg.norm(B@x)**2

def grad_g(A,B,x,g):
    return (2/(np.linalg.norm(B@x))**2)*(A.T@(A@x)-g(A,B,x)*B.T@(B@x))

def prox_h(x, step_size,lambd):
    return np.sign(x) * np.maximum(np.abs(x) - step_size*lambd/2, 0)

def backtracking_line_search(g, prox_h, grad_g, x, lambd,alpha=0.5, beta=0.8):
    t = 1.0
    while g(A,B,prox_h(x - t*grad_g(A,B,x,g),t,lambd)) > g(A,B,x) - alpha * t * np.linalg.norm(grad_g(A,B,x,g))**2:
        t *= beta
    return t

def getfeat(Asol,Bsol,colid,idx):
    w_A = Asol[0:A.shape[1]-1]
    w_B = Bsol[0:B.shape[1]-1]
    w_Aarg = np.abs(w_A).argsort()[::-1]
    w_Barg = np.abs(w_B).argsort()[::-1]
    ida = colid[w_Aarg][0:idx]
    idb = colid[w_Barg][0:idx]
    return set(ida)&set(idb)

def indices(a,b):
  merged = []
  seen = set()
  for num in a:
     if num not in seen:
        merged.append(num)
        seen.add(num)
  for num in b:
     if num not in seen:
        merged.append(num)
        seen.add(num)
  return np.array(merged)

def WNPSVMpredict(X_test,w_1,w_2,b_1,b_2):
    a = np.abs(np.dot(X_test,w_1) + np.ones(X_test.shape[0])*b_1)
    b = np.abs(np.dot(X_test,w_2) + np.ones(X_test.shape[0])*b_2)
    aa = a#/np.linalg.norm(a)
    bb = b#/np.linalg.norm(b)
    y_pred = np.zeros(len(a))
    for i in range(len(a)):
        if aa[i]<=bb[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    return y_pred

def OutputAcc(A,B,Asol,Bsol,qa,qb):
    w_A = Asol[0:A.shape[1]-1]
    w_B = Bsol[0:B.shape[1]-1]
    
    w_Aarg = np.abs(w_A).argsort()[::-1]
    w_Barg = np.abs(w_B).argsort()[::-1]
   
    rho = min(myelbow(w_A, w_B))
    indexx = int(rho)+1
    print(indexx) 
    w_Carg = indices(w_Aarg[0:indexx],w_Barg[0:indexx])
     
    # BBB = np.concatenate((AA[0:qa,w_Aarg][:,0:indexx],BB[0:qb,w_Barg][:,0:indexx]),axis=0) 
    # X_train = StandardScaler().fit_transform(BBB)
    # y_train = np.concatenate((np.array([0]*(qa)),np.array([1]*(qb))))
    # AAA = np.concatenate((AA[qa:AA.shape[0],w_Aarg][:,0:indexx],BB[qb:BB.shape[0],w_Barg][:,0:indexx]),axis=0) 
    # X_test = StandardScaler().fit_transform(AAA)
    # y_test = np.concatenate((np.array([0]*(AA.shape[0]-qa)),np.array([1]*(BB.shape[0]-qb))))
      
    # ## Influenza III
    # BBB = np.concatenate((AA[0:qa,w_Aarg][:,0:indexx],AA[qa:AA.shape[0],w_Barg][:,0:indexx]),axis=0) 
    # X_train = StandardScaler().fit_transform(BBB)
    # y_train = np.concatenate((np.array([0]*(qa)),np.array([1]*(AA.shape[0]-qa))))
    # AAA = np.concatenate((BB[0:qb,w_Aarg][:,0:indexx],BB[qb:BB.shape[0],w_Barg][:,0:indexx]),axis=0) 
    # X_test = StandardScaler().fit_transform(AAA)
    # y_test = np.concatenate((np.array([0]*(qb)),np.array([1]*(BB.shape[0]-qb))))
    
    ## Influenza IV & V
    # BBB = np.concatenate((XYA[:,w_Aarg][:,0:indexx],XYB[:,w_Barg][:,0:indexx]),axis=0) # training class 1 - class 2
    # X_train = StandardScaler().fit_transform(BBB)
    # y_train = np.concatenate((np.array([0]*(XYA.shape[0])),np.array([1]*(XYB.shape[0]))))
    # AAA = np.concatenate((XZA[:,w_Aarg][:,0:indexx],XZB[:,w_Barg][:,0:indexx]),axis=0)
    # X_test = StandardScaler().fit_transform(AAA)
    # y_test = np.concatenate((np.array([0]*(XZA.shape[0])),np.array([1]*(XZB.shape[0]))))
    
    ## Influenza VI
    BBB = np.concatenate((XY[0:qa,w_Aarg][:,0:indexx],XY[0:qb,w_Barg][:,0:indexx]),axis=0) # training class 1 - class 2
    X_train = StandardScaler().fit_transform(BBB)
    y_train = np.concatenate((np.array([0]*qa),np.array([1]*qb)))
    AAA = np.concatenate((XZ[0:cb,w_Aarg][:,0:indexx],XZ[cb:XZ.shape[0],w_Barg][:,0:indexx]),axis=0)
    X_test = StandardScaler().fit_transform(AAA)
    y_test = np.concatenate((np.array([0]*cb),np.array([1]*(XZ.shape[0]-cb))))
      
    model = svm.SVC()
    param_grid = {'C': [1,10,100,1e-2,1e-3,1e-1],
                  'kernel':['linear']}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
      
    clfSVM = svm.SVC(kernel='linear',C=grid_result.best_params_['C']).fit(X_train, y_train)
    y_predNSVM = WNPSVMpredict(X_test,w_A[w_Aarg[0:indexx]],w_B[w_Barg[0:indexx]],Asol[-1],Bsol[-1])
    y_predSVM = clfSVM.predict(X_test)
    print('NSVM:', balanced_accuracy_score(y_test, y_predNSVM))
    print('SVM:', balanced_accuracy_score(y_test, y_predSVM))
    print(confusion_matrix(y_test, y_predSVM))
    
    return balanced_accuracy_score(y_test, y_predSVM),indexx,w_Carg

def myelbow(w_A,w_B):
    x = range(0,len(w_A))
    y = np.sort(np.abs(w_A))[::-1]
    kn_A = KneeFinder(x,y)
    xp,yp = kn_A.find_knee()
    
    mm = range(0,len(w_B))
    my = np.sort(np.abs(w_B))[::-1]
    km_B = KneeFinder(mm,my)
    xm,ym = km_B.find_knee()
    return xp,xm

def plotweights(w_A,w_B,the):
    x = range(0,len(w_A))
    y = np.sort(np.abs(w_A))[::-1]
    kn_A = KneeFinder(x,y)
    xp,yp = kn_A.find_knee()
    
    mm = range(0,len(w_B))
    my = np.sort(np.abs(w_B))[::-1]
    km_B = KneeFinder(mm,my)
    xm,ym = km_B.find_knee()

    fig,ax = plt.subplots(figsize = (12,8))
    plt.plot(np.sort(np.abs(w_A))[::-1],'*b',ms=10)
    #plt.plot(np.sort(np.abs(w_A))[::-1],'--*b',ms=20,lw=5)
    plt.ylabel('sorted weights',fontsize=20)
    plt.xlabel('sorted weight index',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.plot(xp,yp,"or",ms=10)
    legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                          markerfacecolor='blue', markersize=20),
                   Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                          markerfacecolor='red', markersize=20)]
    
    plt.legend(handles=legend_elements,title='{:}'.format(the),title_fontsize='xx-large',fontsize=20)
    for a, b in zip([int(xp)],[round(yp,3)]):
        plt.annotate(f'({a}, {b})', xy=(a, b), xytext=(10,10), fontsize=15, textcoords='offset points', arrowprops=dict())
    
    plt.plot(np.sort(np.abs(w_B))[::-1],'*r',ms=10)
    #plt.plot(np.sort(np.abs(w_B))[::-1],'--*r',ms=20,lw=5)
    plt.ylabel('sorted weights',fontsize=20)
    plt.xlabel('sorted weight index',fontsize=20)
    plt.plot(xm,ym,"ob",ms=10)
    for a, b in zip([int(xm)],[round(ym,3)]):
        plt.annotate(f'({a}, {b})', xy=(a, b), xytext=(5,60), fontsize=15, textcoords='offset points', arrowprops=dict())
    plt.title('{:}'.format(pp),fontsize=20)
    plt.show()
    
def plotpca(CC,the):  
    pca = PCA(n_components=2)
    
    DC = StandardScaler().fit_transform(CC)
    X_pca = pca.fit_transform(DC)
    #labs = np.array([0]*qa+[1]*qb+[2]*(AA.shape[0]-qa)+[3]*(BB.shape[0]-qb)) ### All
    #labs = np.array([0]*qa+[1]*(AA.shape[0]-qa)+[2]*qb+[3]*(BB.shape[0]-qb)) ## Influenza III
    #labs = np.array([0]*qa+[1]*qb+[2]*(XZA.shape[0])+[3]*(XZB.shape[0])) ## Influenza IV & V
    labs = np.array([0]*qa+[1]*qb+[2]*cb+[3]*(XZ.shape[0]-cb)) ## Influenza IV & V
    fig,ax = plt.subplots(figsize = (12,8))
    plt.scatter(X_pca[labs==0,0], X_pca[labs==0,1],color='blue',label='{:}'.format(pp1),marker='o',s=200)
    plt.scatter(X_pca[labs==1,0], X_pca[labs==1,1],color='red',label='{:}'.format(pp2),marker='o',s=200)
    plt.scatter(X_pca[labs==2,0], X_pca[labs==2,1],color='g',label='{:}'.format(pp3),marker='X',s=200)
    plt.scatter(X_pca[labs==3,0], X_pca[labs==3,1],color='k',label='{:}'.format(pp4),marker='X',s=200)
    
    plt.legend(title='{:}'.format(the),title_fontsize='xx-large',fontsize=20)
    
    plt.xlabel('PC1',fontsize=20)
    plt.ylabel('PC2',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.title('{:}'.format(pp),fontsize=20)
    

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def computeavgjac(sets):
    num_sets = len(sets)
    total_similarity = 0
    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            similarity = jaccard_similarity(sets[i], sets[j])
            total_similarity += similarity
    average_similarity = total_similarity / (num_sets * (num_sets - 1) / 2)
    return  average_similarity
    
# data, meta = arff.loadarff('Ovarian.arff')
# Datta = pd.DataFrame(data) # the last column is the class
# Meta = pd.DataFrame(meta)
# #Datta['Class'].value_counts()
# label = Datta['Class'].replace({b'Normal':0,b'Cancer':1})
# Label = label[label.argsort()[::1]] # Sort label for normal and cancer in that order
# Data = Datta.iloc[label.argsort()[::1],:] # Sort the rows of the data
# AA = Data.iloc[0:len(label[label==0]),0:Datta.shape[1]-1].values # Normal class
# BB = Data.iloc[len(label[label==0]):Data.shape[0],0:Datta.shape[1]-1].values # Cancer class
# qa = int(AA.shape[0]*.8)
# qb = int(BB.shape[0]*.8)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data
# pp1 = 'Train: Normal'
# pp2 = 'Train: Cancer'
# pp3 = 'Test: Normal'
# pp4 = 'Test: Cancer'
# pp = 'Ovarian Cancer Dataset'

# Kingry = scipy.io.loadmat('Prostate_GE.mat') 
# Data = Kingry['X']
# label = Kingry['Y'] # This is ordered already
# AA = Data[0:len(label[label==1]),:] # Normal class
# BB = Data[len(label[label==1]):Data.shape[0],:] # Cancer class
# qa = int(AA.shape[0]*.8)
# qb = int(BB.shape[0]*.8)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data
# pp = 'Lung Cancer Dataset'
# pp1 = 'Train: Normal'
# pp2 = 'Train: Cancer'
# pp3 = 'Test: Normal'
# pp4 = 'Test: Cancer'
# pp = 'Prostate Cancer Dataset'

# Kingry = scipy.io.loadmat('SMK_CAN_187.mat') 
# Data = Kingry['X']
# label = Kingry['Y'] # This is ordered already
# AA = Data[0:len(label[label==1]),:] # Without cancer
# BB = Data[len(label[label==1]):Data.shape[0],:] # With cancer
# qa = int(AA.shape[0]*.8)
# qb = int(BB.shape[0]*.8)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data
# pp1 = 'Train: Smokers without Cancer'
# pp2 = 'Train: Smokers with Cancer'
# pp3 = 'Test: Smokers without Cancer'
# pp4 = 'Test: Smokers with Cancer'
# pp = 'Lung Cancer Dataset'

# Kingry = scipy.io.loadmat('GLI_85.mat') 
# data = Kingry['X'] 
# label = Kingry['Y'].flatten() # This is ordered already
# AA = data[0:len(label[label==1]),:] # Normal class
# BB = data[len(label[label==1]):data.shape[0],:] # tumor class
# qa = int(AA.shape[0]*.8)
# qb = int(BB.shape[0]*.8)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data
# pp1 = 'Train: Normal'
# pp2 = 'Train: Tumor'
# pp3 = 'Test: Normal'
# pp4 = 'Test: Tumor'
# pp = 'Glioma Dataset'

# data, meta = arff.loadarff('Leukemia.arff')
# Datta = pd.DataFrame(data)
# Meta = pd.DataFrame(meta)
# #Datta['Class'].value_counts()
# label = Datta['CLASS'].replace({b'ALL':0,b'AML':1})
# Data = Datta.iloc[label.argsort()[::1],:]
# AA = Data.iloc[0:len(label[label==0]),0:Datta.shape[1]-1].values # Normal class
# BB = Data.iloc[len(label[label==0]):Data.shape[0],0:Datta.shape[1]-1].values # tumor class
# qa = int(AA.shape[0]*.8)
# qb = int(BB.shape[0]*.8)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data
# pp1 = 'Train: ALL'
# pp2 = 'Train: AML'
# pp3 = 'Test: ALL'
# pp4 = 'Test: AML'
# pp = 'Leukemia Dataset'

# Kingry = pd.read_csv('gse73072_data.csv') # data matrix
# data = pd.read_csv('gse73072_metadata.csv') # This is the meta data
# d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
# d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
# label1 = data[0:863] # meta data associated with h1n1
# label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
# nu = 0
# za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
# zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
# dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
# dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
# dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
# dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
# AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
# BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
# qa = int(AA.shape[0]*.80) 
# qb = int(BB.shape[0]*.80)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data
# pp1 = 'Train: H1N1 Non-Shedders+Shedders'
# pp2 = 'Train: H3N2 Non-Shedders+Shedders'
# pp3 = 'Test: H1N1 Shedders'
# pp4 = 'Test: H3N2 Shedders'
# pp = 'Influenza I Dataset'

# Kingry = pd.read_csv('gse73072_data.csv') # data matrix
# data = pd.read_csv('gse73072_metadata.csv') # This is the meta data
# d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
# d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
# label1 = data[0:863] # meta data associated with h1n1
# label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
# nu = 0
# za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
# zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
# dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
# dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
# dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
# dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
# AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
# BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
# qa = len(za[za==0]) # getting number of non-shedders in h1n1
# qb = len(zb[zb==0]) # getting number of non-shedders in h3n2
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0) # training data = all non-shedders
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0) # testing data = all shedders
# pp1 = 'Train: H1N1 Non-Shedders'
# pp2 = 'Train: H3N2 Non-Shedders'
# pp3 = 'Test: H1N1 Shedders'
# pp4 = 'Test: H3N2 Shedders'
# pp = 'Influenza II Dataset'

# Kingry = pd.read_csv('gse73072_data.csv') # data matrix
# data = pd.read_csv('gse73072_metadata.csv') # This is the meta data
# d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
# d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
# label1 = data[0:863] # meta data associated with h1n1
# label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
# nu = 0
# za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
# zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
# dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
# dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
# dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
# dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
# AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
# BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
# qa = len(za[za==0]) # getting number of non-shedders in h1n1
# qb = len(zb[zb==0]) # getting number of non-shedders in h3n2
# XY = np.concatenate((AA[0:qa,:],AA[qa:AA.shape[0],:]),axis=0) # training data = h1n1 non-shedders/shedders
# XZ = np.concatenate((BB[0:qb,:],BB[qb:BB.shape[0],:]),axis=0) # testing data = h3n2 non-shedders/shedders
# pp1 = 'Train: H1N1 Non-Shedders'
# pp2 = 'Train: H1N1 Shedders'
# pp3 = 'Test: H3N2 Non-Shedders'
# pp4 = 'Test: H3N2 Shedders'
# pp = 'Influenza III Dataset'

# Kingry = pd.read_csv('gse73072_data.csv') # data matrix
# data = pd.read_csv('gse73072_metadata.csv') # This is the meta data
# d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
# d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
# label1 = data[0:863] # meta data associated with h1n1
# label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
# nu = 0
# za = label1[(label1['time_id']>nu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
# zb = label2[(label2['time_id']>nu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
# dd1 = d1.iloc[label1[(label1['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
# dd2 = d2.iloc[label2[(label2['time_id']>nu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
# dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
# dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
# AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
# BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
# xa = len(za[za==0]) # getting number of non-shedders in h1n1
# xb = len(zb[zb==0]) # getting number of non-shedders in h3n2
# XA = np.concatenate((AA[0:xa,:],BB[0:xb,:]),axis=0) # combining h1n1 and h3n2 non-shedder data
# XB = np.concatenate((AA[xa:AA.shape[0],:],BB[xb:BB.shape[0],:]),axis=0) # combining h1n1 and h3n2 shedder data
# XYA, XZA,_,_ = train_test_split(XA,[0]*XA.shape[0],test_size=0.2, random_state=10) # randomly split non-shedder data into training and testing
# XYB, XZB,_,_ = train_test_split(XB,[0]*XB.shape[0],test_size=0.2, random_state=10) # randomly split shedder data into training and testing
# qa = XYA.shape[0] # number of non-shedder training data
# qb = XYB.shape[0] # number of shedder training data
# XY = np.concatenate((XYA,XYB),axis=0) # training data non-shedders/shedders
# XZ = np.concatenate((XZA,XZB),axis=0) # testing data non-shedders/shedders
# pp1 = 'Train: H1N1+H3N2: Non-Shedders'
# pp2 = 'Train: H1N1+H3N2: Shedders'
# pp3 = 'Test: H1N1+H3N2: Non-Shedders'
# pp4 = 'Test: H1N1+H3N2: Shedders'
# pp = 'Influenza IV Dataset'

# Kingry = pd.read_csv('gse73072_data.csv') # data matrix
# data = pd.read_csv('gse73072_metadata.csv') # This is the meta data
# d1 = Kingry.iloc[0:863,1:Kingry.shape[1]] # h1n1 data
# d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
# label1 = data[0:863] # meta data associated with h1n1
# label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
# nu = 0
# nnu = 24 # upperbound for time_id
# za = label1[(label1['time_id']>0)&(label1['time_id']<=nnu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
# zb = label2[(label2['time_id']>0)&(label2['time_id']<=nnu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
# dd1 = d1.iloc[label1[(label1['time_id']>0)&(label1['time_id']<=nnu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
# dd2 = d2.iloc[label2[(label2['time_id']>0)&(label2['time_id']<=nnu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
# dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
# dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
# AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
# BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
# xa = len(za[za==0]) # getting number of non-shedders in h1n1
# xb = len(zb[zb==0]) # getting number of non-shedders in h3n2
# XA = np.concatenate((AA[0:xa,:],BB[0:xb,:]),axis=0) # combining h1n1 and h3n2 non-shedder data
# XB = np.concatenate((AA[xa:AA.shape[0],:],BB[xb:BB.shape[0],:]),axis=0) # combining h1n1 and h3n2 shedder data
# XYA, XZA,_,_ = train_test_split(XA,[0]*XA.shape[0],test_size=0.2, random_state=10) # randomly split non-shedder data into training and testing
# XYB, XZB,_,_ = train_test_split(XB,[0]*XB.shape[0],test_size=0.2, random_state=10) # randomly split shedder data into training and testing
# qa = XYA.shape[0] # number of non-shedder training data
# qb = XYB.shape[0] # number of shedder training data
# XY = np.concatenate((XYA,XYB),axis=0) # training data non-shedders/shedders
# XZ = np.concatenate((XZA,XZB),axis=0) # testing data non-shedders/shedders
# pp1 = 'Train: H1N1+H3N2: Non-Shedders'
# pp2 = 'Train: H1N1+H3N2: Shedders'
# pp3 = 'Test: H1N1+H3N2: Non-Shedders'
# pp4 = 'Test: H1N1+H3N2: Shedders'
# pp = 'Influenza V Dataset'

Kingry = pd.read_csv('gse73072_data.csv') # data matrix
data = pd.read_csv('gse73072_metadata.csv') # This is the meta data
d1 = Kingry.iloc[477:863,1:Kingry.shape[1]].reset_index(drop=True) # Note h1n1 dee3 data [0:477] excluded for testing
d2 = Kingry.iloc[863:Kingry.shape[0],1:Kingry.shape[1]].reset_index(drop=True) #h3n2 data
label1 = data[477:863].reset_index(drop=True) # meta data associated with h1n1
label2 = data[863:Kingry.shape[0]].reset_index(drop=True) # meta data associated with h3n2
nu = 0
nnu = 24 # upperbound for time_id
za = label1[(label1['time_id']>0)&(label1['time_id']<=nnu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
zb = label2[(label2['time_id']>0)&(label2['time_id']<=nnu)]['shedding'].astype(int) #removed h3n2 controls (time_id<=0), then convert T/F to 1/0
dd1 = d1.iloc[label1[(label1['time_id']>0)&(label1['time_id']<=nnu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
dd2 = d2.iloc[label2[(label2['time_id']>0)&(label2['time_id']<=nnu)].index,:] # Use the indices for time_id>0 to select rows in h3n2
dan1 = np.argsort(za) # get indices for h1n1 shedders(1)/non-shedders(0)
dan2 = np.argsort(zb) # get indices for h3n2 shedders(1)/non-shedders(0)
AA = dd1.iloc[dan1,:].values # Sorting h1n1 with non-shedders up/shedders down
BB = dd2.iloc[dan2,:].values # Sorting h3n2 with non-shedders up/shedders down
xa = len(za[za==0]) # getting number of non-shedders in h1n1
xb = len(zb[zb==0]) # getting number of non-shedders in h3n2
XA = np.concatenate((AA[0:xa,:],BB[0:xb,:]),axis=0) # combining h1n1 and h3n2 non-shedder data
XB = np.concatenate((AA[xa:AA.shape[0],:],BB[xb:BB.shape[0],:]),axis=0) # combining h1n1 and h3n2 shedder data
XY = np.concatenate((XA,XB),axis=0) # training data non-shedders/shedders
qa = XA.shape[0] # number of non-shedder training data
qb = XB.shape[0] # number of shedder training data
c1 = Kingry.iloc[0:477,1:Kingry.shape[1]].reset_index(drop=True) # Note h1n1 dee3 data [0:477] excluded for testing
label1 = data[0:477].reset_index(drop=True) # meta data associated with h1n1
zc = label1[(label1['time_id']>0)&(label1['time_id']<=nnu)]['shedding'].astype(int) #removed h1n1 controls (time_id<=0), then convert T/F to 1/0
cc1 = c1.iloc[label1[(label1['time_id']>0)&(label1['time_id']<=nnu)].index,:] # Use the indices for time_id>0 to select rows in h1n1
can1 = np.argsort(zc) # get indices for h1n1 shedders(1)/non-shedders(0)
XZ = cc1.iloc[can1,:].values # Sorting h1n1 with non-shedders up/shedders down
cb = len(zc[zc==0]) # getting number of non-shedders in h1n1
pp1 = 'Train: H1N1+H3N2: Non-Shedders'
pp2 = 'Train: H1N1+H3N2: Shedders'
pp3 = 'Test: H1N1+H3N2: Non-Shedders'
pp4 = 'Test: H1N1+H3N2: Shedders'
pp = 'Influenza VI Dataset'

# data = pd.read_csv('diabetes.csv')
# ##colid = data.columns
# label = data['Outcome'] # condition
# AA = data.iloc[0:len(label[label==0]),0:data.shape[1]-1].values
# BB = data.iloc[len(label[label==0]):data.shape[0],0:data.shape[1]-1].values
# qa = int(AA.shape[0]*.80)
# qb = int(BB.shape[0]*.80)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0)
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0)
# pp1 = 'Train: No'
# pp2 = 'Train: Yes'
# pp3 = 'Test: No'
# pp4 = 'Test: Yes'
# pp = 'Diabetes Dataset'

# data = pd.read_csv('heart_cleveland.csv')
## colid = data.columns
# label = data['condition'] # condition
# AA = data.iloc[0:len(label[label==0]),0:data.shape[1]-1].values
# BB = data.iloc[len(label[label==0]):data.shape[0],0:data.shape[1]-1].values
# qa = int(AA.shape[0]*.80)
# qb = int(BB.shape[0]*.80)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0)
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0)
# pp1 = 'Train: No'
# pp2 = 'Train: Yes'
# pp3 = 'Test: No'
# pp4 = 'Test: Yes'
# pp = 'Heart Disease Dataset'

# data = pd.read_csv('breast-cancer.csv')
# colid = data.columns[2:data.shape[1]]
# label = data['diagnosis'].replace({'B':0,'M':1})
# AA = data.iloc[0:len(label[label==0]),2:data.shape[1]].values
# BB = data.iloc[len(label[label==0]):data.shape[0],2:data.shape[1]].values
# qa = int(AA.shape[0]*.80)
# qb = int(BB.shape[0]*.80)
# XY = np.concatenate((AA[0:qa,:],BB[0:qb,:]),axis=0)
# XZ = np.concatenate((AA[qa:AA.shape[0],:],BB[qb:BB.shape[0],:]),axis=0)
# pp1 = 'Train: Benign'
# pp2 = 'Train: Malignant'
# pp3 = 'Test: Benign'
# pp4 = 'Test: Malignant'
# pp = 'Breast Cancer Dataset'

XX = StandardScaler().fit_transform(XY)
X1 = XX[0:qa,:]
X2 = XX[qa:XX.shape[0],:]
A = np.hstack((X1,np.ones((X1.shape[0],1))))
B = np.hstack((X2,np.ones((X2.shape[0],1))))
# A = X1
# B = X2

step_size = 0.01 # Step size
lambd1 = 1e-1
lambd2 = 1e-1

#------------------ Lq-PGD-GSVD/SVD/QR/PGD-GSVPSVM--------------------
np.random.seed(10)
x00 = np.random.randn(A.shape[1])
x00 = x00/(np.linalg.norm(x00))  # Initial point

kk = 200
runs = 10
Bal_acc = np.zeros(runs)
idex = np.zeros(runs)
setjac = []
ks = 0
for s in range(runs):
    ks = ks + 1
    A = A[random.sample(range(A.shape[0]),A.shape[0]),:]
    B = B[random.sample(range(B.shape[0]),B.shape[0]),:]
    
    the = '$\ell_1$-PGD-GSVPSVM'
    #step_size = backtracking_line_search(g, prox_h, grad_g, x00, lambd1,alpha=0.5, beta=0.8)
    Apgd = proximal_gradient_descent(grad_g, prox_h, g,A,B,x00, step_size,lambd1,num_iterations=kk)
    #step_size = backtracking_line_search(g, prox_h, grad_g, x00, lambd2,alpha=0.5, beta=0.8)
    Bpgd = proximal_gradient_descent(grad_g, prox_h, g,B,A,x00, step_size,lambd2,num_iterations=kk)
    a_w = Apgd[0:A.shape[1]-1].argsort()[::-1]
    b_w = Bpgd[0:B.shape[1]-1].argsort()[::-1] 
    absw_Apgd = np.sort(np.abs(Apgd[0:A.shape[1]-1]))[::-1]
    absw_Bpgd = np.sort(np.abs(Bpgd[0:B.shape[1]-1]))[::-1]
    plotweights(absw_Apgd,absw_Bpgd,the) 
    Bal_acc[ks-1],idex[ks-1],idexg = OutputAcc(A,B,Apgd,Bpgd,qa,qb)
    # CC = np.concatenate((AA[0:qa,a_w][:,0:1+int(idex[ks-1])],BB[0:qb,b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    # CD = np.concatenate((AA[qa:AA.shape[0],a_w][:,0:1+int(idex[ks-1])],BB[qb:BB.shape[0],b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    # EF = np.concatenate((CC,CD))
    # plotpca(EF,the)
    
    # # Influenza III - PCA
    # CC = np.concatenate((AA[0:qa,a_w][:,0:1+int(idex[ks-1])],AA[qa:AA.shape[0],b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    # CD = np.concatenate((AA[0:qb,a_w][:,0:1+int(idex[ks-1])],BB[qb:BB.shape[0],b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    # EF = np.concatenate((CC,CD))
    # plotpca(EF,the)
    
    # # Influenza IV&V - PCA
    # CC = np.concatenate((XYA[:,a_w][:,0:1+int(idex[ks-1])],XYB[:,b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    # CD = np.concatenate((XZA[:,a_w][:,0:1+int(idex[ks-1])],XZB[:,b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    # EF = np.concatenate((CC,CD))
    # plotpca(EF,the)
    
    # Influenza VI - PCA
    CC = np.concatenate((XY[0:qa,a_w][:,0:1+int(idex[ks-1])],XY[0:qb,b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    CD = np.concatenate((XZ[0:cb,a_w][:,0:1+int(idex[ks-1])],XZ[cb:XZ.shape[0],b_w][:,0:1+int(idex[ks-1])]),axis=0) 
    EF = np.concatenate((CC,CD))
    plotpca(EF,the)
    
    setjac.append(set(idexg))
    
print('avgjac:=',computeavgjac(setjac))
print('avgacc:=',np.sum(Bal_acc)/len(Bal_acc))
print('stdacc:=',np.std(Bal_acc)/len(Bal_acc))
print('feat:=',np.sum(idex)/len(idex))



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
from gsvp_py.solvers.features import indices
from gsvp_py.solvers.plots import myelbow
from gsvp_py.solvers.getting_data import getdata
from gsvp_py.solvers.gridsearch import grid_search
from gsvp_py.solvers.gsvp import *
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import random
import joblib
from gsvp_py.solvers.plots import combined_plots
import pandas as pd

def compute_spec_recall(conf_matrix):
    TN, FP, FN, TP = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return specificity, recall, precision

def WNPSVMpredict(X_test,w_1,w_2,b_1,b_2):
    a = np.abs(np.dot(X_test,w_1) + np.ones(X_test.shape[0])*b_1)
    b = np.abs(np.dot(X_test,w_2) + np.ones(X_test.shape[0])*b_2)
    aa = a/np.linalg.norm(a)
    bb = b/np.linalg.norm(b)
    y_pred = np.zeros(len(a))
    for i in range(len(a)):
        if aa[i]<=bb[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    return y_pred

def OutputAcc(A, X_test, y_test, Asol,Bsol,classifier):
    w_A = Asol[0:A.shape[1]]
    w_B = Bsol[0:A.shape[1]]
    
    w_Aarg = np.abs(w_A).argsort()[::-1]
    w_Barg = np.abs(w_B).argsort()[::-1]
   
    rho = min(myelbow(w_A, w_B))
    indexx = int(rho) + 1
    Carg = indices(w_Aarg[0:indexx],w_Barg[0:indexx]) # merge without repeating
    w_Carg = {'topA':w_Aarg[0:indexx],'topB':w_Barg[0:indexx],'common':set(w_Aarg[0:indexx]).intersection(set(w_Barg[0:indexx])),'merged':Carg}

    if classifier == 'npsvm':
        y_pred = WNPSVMpredict(X_test,w_A,w_B,Asol[-1],Bsol[-1])
    elif classifier == 'elb_npsvm':
        y_pred = WNPSVMpredict(X_test[:,Carg],w_A[w_Aarg][0:len(Carg)],w_B[w_Barg][0:len(Carg)],Asol[-1],Bsol[-1])

    #print(confusion_matrix(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    spec, recall, precision = compute_spec_recall(conf_matrix)

    return spec, recall, precision, balanced_accuracy_score(y_test, y_pred),conf_matrix,indexx,w_Carg
        
def classification(data,thee,classifier,reg,lambd,seed,step_size,**kwargs):
    for datum in [data]:
        X_train,y_train,Xtest,y_test,pp1,pp2,pp3,pp4,pc = getdata(datum)

        scaler =  StandardScaler()
        joblib.dump(scaler, f'scaler_{datum}.pkl')

        XX = scaler.fit_transform(X_train)

        X1 = XX[0:len(y_train[y_train==0]),:]
        X2 = XX[len(y_train[y_train==0]):XX.shape[0],:]
        A = np.hstack((X1,np.ones((X1.shape[0],1))))
        B = np.hstack((X2,np.ones((X2.shape[0],1))))

        X_test = scaler.transform(Xtest)

        np.random.seed(seed)
        x00 = np.random.rand(A.shape[1])
        x00 = x00/(np.linalg.norm(x00))  

        lambd1 = lambd[0]
        lambd2 = lambd[1]

        q = kwargs['q']
        epsa = kwargs['epsa']

        if reg == 'l1':
            Apgd,objvalA,Re_errA,Re_fidA,Re_regA = pgd(A,B,x00, step_size,lambd1,reg)
            Bpgd,objvalB,Re_errB,Re_fidB,Re_regB = pgd(B,A,x00, step_size,lambd2,reg) 
        elif reg == 'lq1':
            Apgd,objvalA,Re_errA,Re_fidA,Re_regA = pgd(A,B,x00, step_size,lambd1,reg,**{'q':q,'epsa':epsa})
            Bpgd,objvalB,Re_errB,Re_fidB,Re_regB = pgd(B,A,x00, step_size,lambd2,reg,**{'q':q,'epsa':epsa})


        np.save('C:/Users/uugob/GSVP/gsvp/solutions/first_sol_{}.npy'.format(datum),Apgd)
        np.save('C:/Users/uugob/GSVP/gsvp/solutions/2nd_sol_{}.npy'.format(datum),Bpgd)

        Spec, Recall, precision, Bal_acc,conf_matrix,idex,idexg = OutputAcc(X_train,X_test,y_test,Apgd,Bpgd,classifier) 

        dic_pgd = {'a':objvalA,'b':objvalB,'c': Apgd,'d': Bpgd,'e':Re_errA,'f':Re_errB,'g':Re_fidA,'h':Re_regA,'i':Re_fidB,'j':Re_regB}
        
        # print('Common Feature Indices :=', idexg['common'])
        # print('Dispaying results for {:}'.format(pc))
        # print('Balanced Acc :=', Bal_acc, 'Specificity :=', Spec, 'Recall :=', Recall, 'Precision :=', precision)
        # print('Elbow points :=', idex) 
        results = []
        results.append({'Elbow': idex, 'Common': len(idexg['common']), 'Dataset ': '{:}'.format(pc).replace('Dataset','').strip(), 'Bal. Acc.': Bal_acc, 'Specificity': Spec, 'Recall': Recall, 'Precision': precision, 'TN': conf_matrix[0][0], 'FP': conf_matrix[0][1], 'FN': conf_matrix[1][0], 'TP': conf_matrix[1][1]})

        results_df = pd.DataFrame(results)

        print(results_df)

        combined_plots(X_train, X_test, y_train, y_test, dic_pgd, pc, pp1, pp2, pp3, pp4,thee,idexg['merged'])
    return idexg,Bal_acc,idex

def OutputAcc_test(X_test, y_test, Asol,Bsol):
    w_A = Asol[0:X_test.shape[1]]
    w_B = Bsol[0:X_test.shape[1]]

    y_pred = WNPSVMpredict(X_test,w_A,w_B,Asol[-1],Bsol[-1])

    #print(confusion_matrix(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    spec, recall, precision = compute_spec_recall(conf_matrix)

    return spec, recall, precision,conf_matrix, balanced_accuracy_score(y_test, y_pred)

def test_model(datum, data, datra):
   
    results = []

    for i in range(len(datum)):
        Asol = np.load(f'C:/Users/uugob/GSVP/gsvp/solutions/first_sol_{datum[i]}.npy') 
        Bsol = np.load(f'C:/Users/uugob/GSVP/gsvp/solutions/2nd_sol_{datum[i]}.npy') 
        Xtest = np.load(f'C:/Users/uugob/GSVP/Testing_data/X_test_{data[i]}.npy')
        y_test = np.load(f'C:/Users/uugob/GSVP/Testing_data/y_test_{data[i]}.npy')
        scaler = joblib.load(f'scaler_{datum[i]}.pkl')
        Xtrain = np.load(f'C:/Users/uugob/GSVP/Training_data/X_train_{data[i]}.npy')
        scaler.fit(Xtrain)

        X_test = scaler.transform(Xtest)

        Spec, Recall, precision, conf_matrix,Bal_acc = OutputAcc_test(X_test, y_test, Asol,Bsol)
        results.append({'Dataset': datra[i], 'Bal. Acc.': Bal_acc, 'Specificity': Spec, 'Recall': Recall, 'Precision': precision, 'TN': conf_matrix[0][0], 'FP': conf_matrix[0][1], 'FN': conf_matrix[1][0], 'TP': conf_matrix[1][1]})

    results_df = pd.DataFrame(results)

    print(results_df)
import numpy as np
import random
from gesp_py.solvers.getting_data import getdata
from gesp_py.solvers.gridsearch import grid_search
from gesp_py.solvers.plots import plotweights_subplots
from sklearn.preprocessing import StandardScaler
from gesp_py.solvers.gsvp import *
from gesp_py.solvers.classifiers import OutputAcc
from gesp_py.solvers.metric import computeavgjac
from gesp_py.solvers.classifiers import best_param

def classification(datum):
        # load data
        X_train,y_train,X_test,y_test,_,_,_,_,_ = getdata(datum)

        # process data
        XX = StandardScaler().fit_transform(X_train)
        X1 = XX[0:len(y_train[y_train==0]),:]
        X2 = XX[len(y_train[y_train==0]):XX.shape[0],:]
        A = np.hstack((X1,np.ones((X1.shape[0],1))))
        B = np.hstack((X2,np.ones((X2.shape[0],1))))

        svmpara = best_param(X_train, y_train)
        print(svmpara)

        # find model parameters via grid-search
        np.random.seed(10)
        x00 = np.random.randn(A.shape[1])
        x00 = x00/(np.linalg.norm(x00))  # Initial point
        step_size = 0.01 # Step size

        lambd1,kk1 = grid_search(A,B,x00,step_size)
        lambd2,kk2 = grid_search(B,A,x00,step_size)

        runs = 10
        Bal_acc = np.zeros(runs)
        idex = np.zeros(runs)
        setjac = []
        ks = 0

        dic_pgd = {}
        for s in range(runs):
            ks = ks + 1
            A = A[random.sample(range(A.shape[0]),A.shape[0]),:]
            B = B[random.sample(range(B.shape[0]),B.shape[0]),:]
            the = '$\ell_1$-PGD-GSVPSVM'
            Apgd,objvalA = pgd(A,B,x00, step_size,lambd1,num_iterations=kk1)
            Bpgd,objvalB = pgd(B,A,x00, step_size,lambd2,num_iterations=kk2)
            Bal_acc[ks-1],idex[ks-1],idexg = OutputAcc(X_train,y_train,X_test,y_test,Apgd,Bpgd,svmpara,method='linearsvm')
            dic_pgd[s] = {'a':objvalA,'b':objvalB,'c': Apgd,'d': Bpgd}
            #plotpca(AA,BB,the,pp,pp1,pp2,pp3,pp4,pcx='any')
            setjac.append(set(idexg))
        print('Dispaying results for {:}'.format(datum))
        print('avgjac:=',computeavgjac(setjac))
        print('avgacc:=',np.sum(Bal_acc)/len(Bal_acc))
        #print('stdacc:=',np.std(Bal_acc)/len(Bal_acc))
        print('feat:=',np.sum(idex)/len(idex))

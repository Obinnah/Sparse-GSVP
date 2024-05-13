import numpy as np
from gsvp_py.solvers.getting_data import getdata
from gsvp_py.solvers.gridsearch import grid_search
from sklearn.preprocessing import StandardScaler
from gsvp_py.solvers.gsvp import *
from gsvp_py.solvers.plots import combined_plots

def plotweights(datum,reg,thee,step_size,seed,**kwargs):
    for pp in [datum]:
    
        X_train,y_train,X_test,y_test,pp1,pp2,pp3,pp4,pc = getdata(pp)
        XX = StandardScaler().fit_transform(X_train)
        X1 = XX[0:len(y_train[y_train==0]),:]
        X2 = XX[len(y_train[y_train==0]):XX.shape[0],:]
        A = np.hstack((X1,np.ones((X1.shape[0],1))))
        B = np.hstack((X2,np.ones((X2.shape[0],1))))
       
        np.random.seed(seed)
        x00 = np.random.rand(A.shape[1])
        x00 = x00/(np.linalg.norm(x00))
        
        if reg == 'l1':
            lambd1,kk1 = grid_search(A,B,x00,step_size,pp,reg)
            lambd2,kk2 = grid_search(B,A,x00,step_size,pp,reg)
        elif reg == 'lq1':
            lambd1,kk1 = grid_search(A,B,x00,step_size,pp,reg,**kwargs)
            lambd2,kk2 = grid_search(B,A,x00,step_size,pp,reg,**kwargs)
        elif reg == 'l12':
            lambd1,kk1,beta1 = grid_search(A,B,x00,step_size,pp,reg)
            lambd2,kk2,beta2 = grid_search(B,A,x00,step_size,pp,reg)
        elif reg == 'lq12':
            lambd1,kk1,beta1 = grid_search(A,B,x00,step_size,pp,reg,**kwargs)
            lambd2,kk2,beta2 = grid_search(B,A,x00,step_size,pp,reg,**kwargs)
            #print(f'beta1:{beta1},beta2:{beta2},kk1:{kk1},kk2:{kk2},lambd1:{lambd1},lambd2:{lambd2}')
    
        if reg == 'l1':
            Apgd,objvalA,Re_errA,Re_fidA,Re_regA = pgd(A,B,x00, step_size,kk1,lambd1,reg)
            Bpgd,objvalB,Re_errB,Re_fidB,Re_regB = pgd(B,A,x00, step_size,kk2,lambd2,reg)
            print('2-norm of solutions', np.linalg.norm(Apgd),np.linalg.norm(Bpgd), 'size of training matrices',A.shape,B.shape,'iteration', kk1,kk2, 'regpara',lambd1,lambd2)
        elif reg == 'l12':
            Apgd,objvalA,Re_errA,Re_fidA,Re_regA = pgd(A,B,x00, step_size,kk1,lambd1,reg,**{'beta':beta1})
            Bpgd,objvalB,Re_errB,Re_fidB,Re_regB = pgd(B,A,x00, step_size,kk2,lambd2,reg,**{'beta':beta2})
        elif reg == 'lq1':
            q = kwargs['q']
            epsa = kwargs['epsa']
            Apgd,objvalA,Re_errA,Re_fidA,Re_regA = pgd(A, B, x00, step_size,kk1,lambd1,reg,**kwargs)
            Bpgd,objvalB,Re_errB,Re_fidB,Re_regB = pgd(B, A, x00, step_size,kk2,lambd2,reg,**{'q':q,'epsa':epsa})
        elif reg == 'lq12':
            q = kwargs['q']
            epsa = kwargs['epsa']
            Apgd,objvalA,Re_errA,Re_fidA,Re_regA = pgd(A, B, x00, step_size,kk1,lambd1,reg,**{'beta':beta1,'q':q,'epsa':epsa})
            Bpgd,objvalB,Re_errB,Re_fidB,Re_regB = pgd(B, A, x00, step_size,kk2,lambd2,reg,**{'beta':beta2,'q':q,'epsa':epsa}) 
        dic_pgd = {'a':objvalA,'b':objvalB,'c': Apgd,'d': Bpgd,'e':Re_errA,'f':Re_errB,'g':Re_fidA,'h':Re_regA,'i':Re_fidB,'j':Re_regB}
        combined_plots(X_train, X_test, y_train, y_test, dic_pgd, pc, pp1, pp2, pp3, pp4,thee)

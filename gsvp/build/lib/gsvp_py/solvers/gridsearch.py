import numpy as np
from gsvp_py.solvers.gsvp import *

def grid_search(A,B,x_initial,step_size,datum,reg,**kwargs):

    if datum in ['Diabetes Dataset','Heart Disease Dataset','Breast Cancer Dataset']:
        lambd_values = np.linspace(0.1,0.2,3)
        num_iterations_values = [5,10,15,20,25,30,35,40,50,100,200,250,300]  
        beta_values = np.linspace(0.1,10,10)
        if reg in ['lq1']:
            q = kwargs['q']
            epsa = kwargs['epsa']

    elif datum in ['Influenza I']:
        lambd_values = np.linspace(0.1,0.2,3,endpoint=0) 
        num_iterations_values = [5,10,15,20,25,30]  
        if reg in ['lq1']:
            q = kwargs['q']
            epsa = kwargs['epsa']

    elif datum in ['Ovarian Cancer Dataset', 'Prostate Cancer Dataset']:
        lambd_values = np.linspace(0.1,1,6,endpoint=0)
        num_iterations_values = [50,100,150,200]  
        if reg in ['lq1']:
            q = kwargs['q']
            epsa = kwargs['epsa']

    best_score = float('inf')  
    best_lambd = None
    best_num_iterations = None 
    best_beta = None

    for lambd in lambd_values:
        for num_iterations in num_iterations_values:
                if reg in ['l1','lq1']:
                    if reg == 'l1':
                        x,_,_,_,_= pgd(A, B, x_initial, step_size,num_iterations,lambd,reg)
                        obj_value = (np.linalg.norm(A@x)**2/np.linalg.norm(B@x)**2)+lambd*np.linalg.norm(x,ord=1)
                    elif reg == 'lq1':
                        x, _,_,_,_= pgd(A, B, x_initial, step_size,num_iterations,lambd,reg,**{'q':q,'epsa':epsa})
                        obj_value = (np.linalg.norm(A@x)**2/np.linalg.norm(B@x)**2)+lambd*np.linalg.norm(threshold(x,epsa,q)*x)**2
                    if obj_value < best_score:
                        best_score = obj_value
                        best_lambd = lambd
                        best_num_iterations = num_iterations
                        
    if reg in ['l1','lq1']:
        # print(f"best_lambd: {best_lambd}, best_num_iterations: {best_num_iterations}, obj_value: {best_score}")
        return best_lambd, best_num_iterations

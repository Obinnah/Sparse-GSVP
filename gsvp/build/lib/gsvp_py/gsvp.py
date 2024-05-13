import numpy as np

def pgd(A,B,x_initial, step_size,num_iterations,lambd,reg,**kwargs):
    obj_value = []
    rel_error = []
    x0 = x_initial

    if reg == 'l1':
        obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(x0,ord=1)
    elif reg == 'l12':
        beta = kwargs['beta']
        obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(x0,ord=1)+beta*np.linalg.norm(x0)**2
    elif reg == 'lq1':
        q = kwargs['q']
        epsa = kwargs['epsa']
        obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(threshold(x0,epsa,q)*x0)
    elif reg == 'lq12':
        q = kwargs['q']
        beta = kwargs['beta']
        epsa = kwargs['epsa']
        obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(threshold(x0,epsa,q)*x0)+beta*np.linalg.norm(x0)**2 
    obj_value.append(obj_val)
   
    for iteration in range(num_iterations):
        gradient = grad_g(A,B,x0,g)
        if reg == 'l1':
            x = prox_h(x0 - step_size * gradient, step_size,lambd,reg)
        elif reg == 'l12':
            x = 1/(1+(step_size*beta))*prox_h(x0 - step_size * gradient, step_size,lambd,reg)
        elif reg in  ['lq12','lq1']:
            x = prox_h(x0 - step_size * gradient, step_size,lambd,reg,**kwargs)
        
        if reg == 'l1':
            obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(x0,ord=1)
        elif reg == 'l12':
            obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(x0,ord=1)+beta*np.linalg.norm(x0)**2
        elif reg == 'lq1':
            obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(threshold(x0,epsa,q)*x0)
        elif reg == 'lq12':
            obj_val = (np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2)+lambd*np.linalg.norm(threshold(x0,epsa,q)*x0)+beta*np.linalg.norm(x0)**2 
        obj_value.append(obj_val)

        error = np.linalg.norm(x-x0)/np.linalg.norm(x0)
        rel_error.append(error)

        #if  error < 1e-5: #iteration >=25 or error > min(rel_error[-20:]):
        if  iteration >30 and error > min(rel_error[-20:]): 
            #print('condition satisfied')
            break 
        else:
            x0 = x
            
    return x, np.array(obj_value),np.array(rel_error)


def threshold(x,epsa,q):
    return (x**2 + epsa**2)**((q-2)/2)

def g(A,B,x):
    return np.linalg.norm(A@x)**2/np.linalg.norm(B@x)**2

def grad_g(A,B,x,g):
    return (2/(np.linalg.norm(B@x))**2)*(A.T@(A@x)-g(A,B,x)*B.T@(B@x))

def prox_h(x, step_size,lambd,reg,**kwargs):
    if reg in ['l1','l12']:
        return np.sign(x) * np.maximum(np.abs(x) - step_size*lambd/2, 0)
    elif reg in ['lq12','lq1']:
        epsa = kwargs['epsa']
        q = kwargs['q']
        D = threshold(x,epsa,q)
        I = np.ones(D.shape[0])
        if reg == 'lq12':
            beta = kwargs['beta']
            return np.multiply(1/((((1+step_size*beta)*I)+step_size*lambd*D)),x)
        elif reg == 'lq1':
            return np.multiply(1/(I+step_size*lambd*D),x)
    


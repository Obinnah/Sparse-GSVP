import numpy as np

def pgd(A,B,x_initial, step_size,lambd,reg,**kwargs):
    obj_value = []
    rel_error = []
    fidelity = []
    regular = []
    x0 = x_initial
    if reg == 'l1':
        fid0 = np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2
        reg0 =  lambd*np.linalg.norm(x0,ord=1)
        obj_val0 = fid0 + reg0
    elif reg == 'lq1':
        q = kwargs['q']
        epsa = kwargs['epsa']
        fid0 = np.linalg.norm(A@x0)**2/np.linalg.norm(B@x0)**2
        reg0 = lambd*np.linalg.norm(threshold(x0,epsa,q)*x0)**2
        obj_val0 = fid0 + reg0
        
    obj_value.append(obj_val0)
    fidelity.append(fid0)
    regular.append(reg0)

    while len(rel_error) < 10000:
        gradient = grad_g(A,B,x0)
        if reg in ['l1']:
            x = prox_h(x0 - step_size * gradient, step_size,lambd,reg)
            fid = np.linalg.norm(A@x)**2/np.linalg.norm(B@x)**2
            regg =  lambd*np.linalg.norm(x,ord=1)
            obj_val = fid + regg
        elif reg in ['lq1']:
            x = prox_h(x0 - step_size * gradient, step_size,lambd,reg,**kwargs)
            fid = np.linalg.norm(A@x)**2/np.linalg.norm(B@x)**2
            regg = lambd*np.linalg.norm(threshold(x0,epsa,q)*x)**2
            obj_val = fid + regg

        obj_value.append(obj_val)
        fidelity.append(fid)
        regular.append(regg)

        error = np.linalg.norm(obj_val-obj_val0)/np.linalg.norm(obj_val0)
        #error = np.linalg.norm(x-x0)/np.linalg.norm(x0)
        rel_error.append(error)
        if  error < 1e-4:
            print('condition satisfied')
            break 
        else:
            obj_val0 = obj_val
            x0 = x

    return x, np.array(obj_value),np.array(rel_error),np.array(fidelity),np.array(regular)


def threshold(x,epsa,q):
    return (x**2 + epsa**2)**((q-2)/4)

def g(A,B,x): 
    return np.linalg.norm(A@x)**2/np.linalg.norm(B@x)**2

def grad_g(A,B,x):
    return (2/(np.linalg.norm(B@x))**2)*(A.T@(A@x)-g(A,B,x)*B.T@(B@x))

def prox_h(x, step_size,lambd,reg,**kwargs):
    if reg in ['l1']:
        return np.sign(x) * np.maximum(np.abs(x) - step_size*lambd/2, 0)
    elif reg in ['lq1']:
        epsa = kwargs['epsa']
        q = kwargs['q']
        D = threshold(x,epsa,q)**2
        I = np.ones(D.shape[0])
        return np.multiply(1/(I+step_size*lambd*D),x)
    



#  error = np.linalg.norm(x-x0)/np.linalg.norm(x0)
# (len(rel_error)>200 and error > min(rel_error[-10:]))
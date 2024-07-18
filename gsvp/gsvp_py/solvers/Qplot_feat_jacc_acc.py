import numpy as np
import matplotlib.pyplot as plt
from gsvp_py.solvers.classifiers import classification
from gsvp_py.solvers.metric import computeavgjac
from gsvp_py.solvers.features import top_features
from gsvp_py.solvers.getting_data import getdata


def Lqplots(setq,the,lambd,seed,pp,reg,classifier,headings,epsa,step_size,sep,**kwargs):

    Bal_acc = np.zeros((len(setq),1))
    idex = np.zeros((len(setq),1))
    jac = []
    
    s = 0
    for q in setq:
        print('q :=', q)
        s = s + 1
        
        if reg in ['lq1','lq12']:
            setjac,Bal_acc[s-1],idex[s-1] = classification(pp,the,classifier,reg,lambd,seed,step_size,sep,**{'q':q,'epsa':epsa})
            top_features(f'heading_{headings}',setjac,**kwargs)
            jac.append(set(setjac['merged']))
        else:
            raise ValueError("Invalid regularization term. Please use 'lq1'")
    print('Jaccard Similarity Index for $q$ :=', computeavgjac(jac))
    
    fig, axs = plt.subplots(1, 1, figsize=(18, 5))

    # Plot Avg. Bal. Acc. Scores
    # axs[0].plot(Bal_acc, '-*r', lw=5, markersize=20, label=the)
    # axs[0].set_ylabel('Bal. Acc. Scores', fontsize=16)
    # axs[0].set_xlabel('$q$', fontsize=20)
    # axs[0].set_xticks(range(len(setq)))  # Adjusted line
    # axs[0].set_xticklabels(setq, rotation=60, fontsize=14)
    # axs[0].tick_params(axis='y', labelsize=14)
    # axs[0].legend(fontsize=14)
    
    # Plot Avg. Genes
    axs.plot(idex, '-*b', lw=5, markersize=20, label=the)
    axs.set_xlabel('$q$', fontsize=20)
    axs.set_xticks(range(len(setq)))  # Adjusted line
    axs.set_xticklabels(setq, rotation=60, fontsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs.set_ylabel('Features', fontsize=16)
    axs.legend(fontsize=14)

    _,_,_,_,_,_,_,_,pc = getdata(pp)
    plt.suptitle('{:}'.format(pc), fontsize=16)

    plt.tight_layout()
    plt.show()


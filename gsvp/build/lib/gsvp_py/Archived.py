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
    



def classification(data,classifier,reg,runs,step_size,**kwargs):
    for datum in [data]:
    #for datum in ['Prostate Cancer Dataset']:
        X_train,y_train,X_test,y_test,_,_,_,_,_ = getdata(datum)

        # process data
        XX = StandardScaler().fit_transform(X_train)
        X1 = XX[0:len(y_train[y_train==0]),:]
        X2 = XX[len(y_train[y_train==0]):XX.shape[0],:]
        A = np.hstack((X1,np.ones((X1.shape[0],1))))
        B = np.hstack((X2,np.ones((X2.shape[0],1))))

        if A.shape[1]<=30:
            para = best_param(X_train, y_train,classifier)

        Bal_acc = np.zeros(runs)
        idex = np.zeros(runs)
        setjac = []
        ks = 0
        dic_pgd = {}
        for s in range(runs):
            ks = ks + 1    

            np.random.seed(s)
            x00 = np.random.rand(A.shape[1])
            x00 = x00/(np.linalg.norm(x00))  # Initial point

            if reg == 'l1':
                lambd1,kk1 = grid_search(A,B,x00,step_size,datum,reg)
                lambd2,kk2 = grid_search(B,A,x00,step_size,datum,reg)
            elif reg == 'lq1':
                lambd1,kk1 = grid_search(A,B,x00,step_size,datum,reg,**kwargs)
                lambd2,kk2 = grid_search(B,A,x00,step_size,datum,reg,**kwargs)
                q = kwargs['q']
                epsa = kwargs['epsa']
            elif reg == 'l12':
                lambd1,kk1,beta1 = grid_search(A,B,x00,step_size,datum,reg)
                lambd2,kk2,beta2 = grid_search(B,A,x00,step_size,datum,reg)
            elif reg == 'lq12':
                lambd1,kk1,beta1 = grid_search(A,B,x00,step_size,datum,reg,**kwargs)
                lambd2,kk2,beta2 = grid_search(B,A,x00,step_size,datum,reg,**kwargs)
                q = kwargs['q']
                epsa = kwargs['epsa']

            A = A[random.sample(range(A.shape[0]),A.shape[0]),:]
            B = B[random.sample(range(B.shape[0]),B.shape[0]),:]
            if reg == 'l1':
                Apgd,objvalA,_ = pgd(A,B,x00, step_size,kk1,lambd1,reg)
                Bpgd,objvalB,_ = pgd(B,A,x00, step_size,kk2,lambd2,reg)
            elif reg == 'l12':
                Apgd,objvalA,_ = pgd(A,B,x00, step_size,kk1,lambd1,reg,**{'beta':beta1})
                Bpgd,objvalB,_ = pgd(B,A,x00, step_size,kk2,lambd2,reg,**{'beta':beta2})
            elif reg == 'lq1':
                Apgd,objvalA,_ = pgd(A,B,x00, step_size,kk1,lambd1,reg,**{'q':q,'epsa':epsa})
                Bpgd,objvalB,_ = pgd(B,A,x00, step_size,kk2,lambd2,reg,**{'q':q,'epsa':epsa})
            elif reg == 'lq12':
                Apgd,objvalA,_ = pgd(A,B,x00, step_size,kk1,lambd1,reg,**{'beta':beta1,'q':q,'epsa':epsa})
                Bpgd,objvalB,_ = pgd(B,A,x00, step_size,kk2,lambd2,reg,**{'beta':beta2,'q':q,'epsa':epsa})
            if A.shape[1]>30:
                para = best_para_selected_features(X_train,y_train,Apgd,Bpgd,classifier)
            Bal_acc[ks-1],idex[ks-1],idexg = OutputAcc(X_train,y_train,X_test,y_test,Apgd,Bpgd,classifier,para) 
            dic_pgd[s] = {'a':objvalA,'b':objvalB,'c': Apgd,'d': Bpgd}
            setjac.append(set(idexg))
        
        print('Dispaying results for {:}'.format(datum))
        print('avgJac :=',computeavgjac(setjac))
        print('avgAcc :=',np.sum(Bal_acc)/len(Bal_acc))
        print('avgStd :=',np.std(Bal_acc)/len(Bal_acc))
        print('avgFeat :=',np.sum(idex)/len(idex)) # idex is the min of the elbow points

        return setjac,Bal_acc,idex
    


    def Lqplots(setq,runs,labels,pp,reg,classifier,headings,epsa,step_size):

    Bal_acc = np.zeros((len(setq),runs))
    idex = np.zeros((len(setq),runs))
    jac = np.zeros((len(setq),1))

    s = 0
    for q in setq:
        print('q :=', q)
        s = s + 1
        setjac = []

        if reg in ['lq1','lq12']:
            setjac,Bal_acc[s-1,:],idex[s-1,:] = classification(pp,classifier,reg,runs,step_size,**{'q':q,'epsa':epsa})
            top_features(f'heading_{headings}',setjac)
        else:
            raise ValueError("Invalid regularization term. Please use 'lq1' or 'lq12'.")

        jac[s-1] = computeavgjac(setjac)

    avgs = np.zeros(len(setq))
    avgsstd = np.zeros(len(setq))
    avgsfeat = np.zeros(len(setq))

    for j in range(len(setq)):
        print('avgacc:=',np.sum(Bal_acc[j,:])/len(Bal_acc[j,:]))
        avgsstd[j] = np.std(Bal_acc[j,:])/len(Bal_acc[j,:])
        avgsfeat[j] = np.sum(idex[j,:])/len(idex[j,:])
        avgs[j] = np.sum(Bal_acc[j,:])/len(Bal_acc[j,:])
        
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Avg. Bal. Acc. Scores
    axs[0].plot(avgs, '-*r', lw=5, markersize=20, label=labels)
    axs[0].set_ylabel('Avg. Bal. Acc. Scores', fontsize=16)
    axs[0].set_xlabel('$q$', fontsize=20)
    axs[0].set_xticks(range(len(setq)))  # Adjusted line
    axs[0].set_xticklabels(setq, rotation=60, fontsize=14)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].legend(fontsize=14)
    #axs[0].set_title('{:}'.format(pp), fontsize=20)

    # Plot Average Jaccard Similarity Index
    axs[1].plot(jac, '-*g', lw=5, markersize=20, label=labels)
    axs[1].set_ylabel('Average Jaccard Similarity Index', fontsize=16)
    axs[1].set_xlabel('$q$', fontsize=20)
    axs[1].set_xticks(range(len(setq)))  # Adjusted line
    axs[1].set_xticklabels(setq, rotation=60, fontsize=14)
    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].legend(fontsize=14)
    #axs[1].set_title('{:}'.format(pp), fontsize=20)

    # Plot Avg. Genes
    axs[2].plot(avgsfeat, '-*b', lw=5, markersize=20, label=labels)
    axs[2].set_xlabel('$q$', fontsize=20)
    axs[2].set_xticks(range(len(setq)))  # Adjusted line
    axs[2].set_xticklabels(setq, rotation=60, fontsize=14)
    axs[2].tick_params(axis='y', labelsize=14)
    axs[2].set_ylabel('Avg. Features', fontsize=16)
    axs[2].legend(fontsize=14)
    #axs[2].set_title('{:}'.format(pp), fontsize=20)

    plt.suptitle('{:}'.format(pp), fontsize=20)

    plt.tight_layout()
    plt.show()

from collections import Counter
def top_features(label,runs_list):
    feature_list = np.load(f'C:/Users/uugob/GESP-PY/dataheads/{label}.npy')
    all_elements = [element for subset in runs_list for element in subset]
    element_counts = Counter(all_elements)
    top_features_counts = element_counts.most_common(10)
    result_df = pd.DataFrame(top_features_counts, columns=['Feature Index', 'Occurrences'])
    result_df['Top Features'] = [feature_list[idx] for idx, _ in top_features_counts]
    result_df = result_df[['Top Features', 'Feature Index', 'Occurrences']]
    print(tabulate(result_df, headers='keys', tablefmt='fancy_grid', showindex=False))

import matplotlib.lines as mlines
def plotweights(datum,reg,thee,step_size,runs,**kwargs):
    # ['Diabetes Dataset','Heart Disease Dataset','Breast Cancer Dataset','Ovarian Cancer Dataset', 'Prostate Cancer Dataset', 'Glioma Dataset',
    # 'Lung cancer Dataset','Leukemia Dataset', 'Influenza I', 'Influenza II', 'Influenza III','Influenza IV', 'Influenza V','Influenza VI',
    # 'depth_results_immature_20230706']:
        
    for pp in [datum]:
    
        X_train,y_train,X_test,y_test,pp1,pp2,pp3,pp4,_ = getdata(pp)
        XX = StandardScaler().fit_transform(X_train)
        #XX = X_train
        X1 = XX[0:len(y_train[y_train==0]),:]
        X2 = XX[len(y_train[y_train==0]):XX.shape[0],:]
        A = np.hstack((X1,np.ones((X1.shape[0],1))))
        B = np.hstack((X2,np.ones((X2.shape[0],1))))
        # A = X1
        # B = X2

        ks = 0
        dic_pgd = {}

        for s in range(runs):
            ks = ks + 1
            np.random.seed(s)
            x00 = np.random.randn(A.shape[1])
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
                Apgd,objvalA,Re_errA = pgd(A,B,x00, step_size,kk1,lambd1,reg)
                Bpgd,objvalB,Re_errB = pgd(B,A,x00, step_size,kk2,lambd2,reg)
                print('2-norm of solutions', np.linalg.norm(Apgd),np.linalg.norm(Bpgd), 'size of training matrices',A.shape,B.shape,'iteration', kk1,kk2, 'regpara',lambd1,lambd2)
            elif reg == 'l12':
                Apgd,objvalA,Re_errA = pgd(A,B,x00, step_size,kk1,lambd1,reg,**{'beta':beta1})
                Bpgd,objvalB,Re_errB = pgd(B,A,x00, step_size,kk2,lambd2,reg,**{'beta':beta2})
            elif reg == 'lq1':
                q = kwargs['q']
                epsa = kwargs['epsa']
                Apgd,objvalA,Re_errA = pgd(A, B, x00, step_size,kk1,lambd1,reg,**kwargs)
                Bpgd,objvalB,Re_errB = pgd(B, A, x00, step_size,kk2,lambd2,reg,**{'q':q,'epsa':epsa})
            elif reg == 'lq12':
                q = kwargs['q']
                epsa = kwargs['epsa']
                Apgd,objvalA,Re_errA = pgd(A, B, x00, step_size,kk1,lambd1,reg,**{'beta':beta1,'q':q,'epsa':epsa})
                Bpgd,objvalB,Re_errB = pgd(B, A, x00, step_size,kk2,lambd2,reg,**{'beta':beta2,'q':q,'epsa':epsa}) 
            dic_pgd[s] = {'a':objvalA,'b':objvalB,'c': Apgd,'d': Bpgd,'e':Re_errA,'f':Re_errB}
        plotweights_subplots(A, dic_pgd, pp, pp1, pp2,thee)
        plot_obj_values(dic_pgd,pp,pp1, pp2,thee)
        plot_rel_err(dic_pgd,pp,pp1, pp2,thee)
        plotpca(X_train,X_test,y_train,y_test,dic_pgd,pp,pp1,pp2,pp3,pp4,thee)


def plotpca(X_train, X_test, y_train, y_test, dic_pgd, pp, pp1, pp2, pp3, pp4,thee):
    A = X_train
    B = X_test
    qa = len(y_train[y_train == 0])
    qb = len(y_test[y_test == 0])

    num_shuffles = len(dic_pgd)
    num_cols = 5
    num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 10))
    fig.suptitle(pp, fontsize=20)

    for shuffle in range(num_shuffles):
        Apgd = dic_pgd[shuffle]['c']
        Bpgd = dic_pgd[shuffle]['d']
        row_idx = shuffle // num_cols
        col_idx = shuffle % num_cols
        ax = axes[row_idx, col_idx]
        a_w = np.abs(Apgd[0:A.shape[1]]).argsort()[::-1]
        b_w = np.abs(Bpgd[0:A.shape[1]]).argsort()[::-1]
        rho = min(myelbow(Apgd[:-1],Bpgd[:-1]))
        indexx = int(rho) + 1
        CC = np.concatenate((A[0:qa, a_w][:, 0:indexx], A[qa:A.shape[0], b_w][:, 0:indexx]), axis=0)
        CD = np.concatenate((B[0:qb, a_w][:, 0:indexx], B[qb:B.shape[0], b_w][:, 0:indexx]), axis=0)
        EF = np.concatenate((CC, CD))
        pca = PCA(n_components=2)
        DC = StandardScaler().fit_transform(EF)
        X_pca = pca.fit_transform(DC)
        labs = np.array([0] * qa  + [1] * (A.shape[0] - qa) + [2]*qb + [3] * (B.shape[0] - qb))
        ax.scatter(X_pca[labs == 0, 0], X_pca[labs == 0, 1], color='blue', label='{:}'.format(pp1), marker='o', s=200)
        ax.scatter(X_pca[labs == 1, 0], X_pca[labs == 1, 1], color='red', label='{:}'.format(pp2), marker='o', s=200)
        ax.scatter(X_pca[labs == 2, 0], X_pca[labs == 2, 1], color='g', label='{:}'.format(pp3), marker='X', s=200)
        ax.scatter(X_pca[labs == 3, 0], X_pca[labs == 3, 1], color='k', label='{:}'.format(pp4), marker='X', s=200)
        ax.legend(title='{:}'.format(thee), title_fontsize='x-large', fontsize=16)
        ax.set_xlabel('PC1', fontsize=16)
        ax.set_ylabel('PC2', fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_title('Shuffle = {:}'.format(shuffle+1), fontsize=16)
        #ax.set_title('{:}'.format(pp), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def plotweights_subplots(A, dic_pgd, pp, pp1, pp2,thee):
    num_shuffles = len(dic_pgd)
    num_cols = 5
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 10))
    fig.suptitle(pp, fontsize=20)
    for shuffle in range(num_shuffles):
        Apgd = dic_pgd[shuffle]['c']
        Bpgd = dic_pgd[shuffle]['d']
        row_idx = shuffle // num_cols
        col_idx = shuffle % num_cols
        ax = axes[row_idx, col_idx]
        m1, n1 = A.shape
        w_A = np.sort(np.abs(Apgd[:-1]))[::-1]  
        w_B = np.sort(np.abs(Bpgd[:-1]))[::-1]  
        x = range(0, len(w_A))
        kn_A = KneeFinder(x, w_A)
        xp, yp = kn_A.find_knee()
        mm = range(0, len(w_B))
        km_B = KneeFinder(mm, w_B)
        xm, ym = km_B.find_knee()
        if m1 < n1:
            ax.plot(w_A, '-*b', ms=10)
        else:
            ax.plot(w_A, '--*b', ms=20, lw=5)
        ax.set_ylabel('sorted weights', fontsize=16)
        ax.set_xlabel('sorted weight index', fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.plot(xp, yp, "or", ms=10)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                   markerfacecolor='blue', markersize=20),
                           Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                   markerfacecolor='red', markersize=20)]

        ax.legend(handles=legend_elements, title='{:}'.format(thee), title_fontsize='x-large', fontsize=16)
        for a, b in zip([int(xp)], [round(yp, 3)]):
            ax.annotate(f'({a+1}, {b})', xy=(a, b), xytext=(10, 10), fontsize=15, textcoords='offset points', arrowprops=dict())
        if m1 < n1:
            ax.plot(w_B,'-*r', ms=10)
        else:
            ax.plot(w_B, '--*r', ms=20, lw=5)
        ax.set_ylabel('Sorted Weights', fontsize=16)
        ax.set_xlabel('Sorted Weight index', fontsize=16)
        ax.plot(xm, ym, "ob", ms=10)
        for a, b in zip([int(xm)], [round(ym, 3)]):
            ax.annotate(f'({a+1}, {b})', xy=(a, b), xytext=(5, 100), fontsize=15, textcoords='offset points', arrowprops=dict())
        ax.set_title('Shuffle = {:}'.format(shuffle+1), fontsize=16)
    # Adjust layout to prevent clipping of ylabel
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_obj_values(dic_pgd,pp,pp1, pp2,thee):
    num_shuffles = 10
    num_cols = 5
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 10))
    fig.suptitle(pp, fontsize=20)
    for shuffle in range(num_shuffles):
        role_AB = dic_pgd[shuffle]['a']
        role_BA = dic_pgd[shuffle]['b']
        row_idx = shuffle // num_cols
        col_idx = shuffle % num_cols
        ax = axes[row_idx, col_idx]
        ax.plot(np.arange(1, len(role_AB) + 1), role_AB, '-*r', lw=10, markersize=20)
        ax.plot(np.arange(1, len(role_BA) + 1), role_BA, '-*b', lw=10, markersize=20)
        ax.set_xlabel('Iterations', fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel('Objective Function', fontsize=16)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                   markerfacecolor='blue', markersize=20),
                           Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                   markerfacecolor='red', markersize=20)]

        ax.legend(handles=legend_elements, title='{:}'.format(thee), title_fontsize='x-large', fontsize=16)
        ax.set_title('Shuffle = {:}'.format(shuffle+1), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_rel_err(dic_pgd,pp,pp1, pp2,thee):
    num_shuffles = 10
    num_cols = 5
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 10))
    fig.suptitle(pp, fontsize=20)
    for shuffle in range(num_shuffles):
        role_AB = dic_pgd[shuffle]['e']
        role_BA = dic_pgd[shuffle]['f']
        row_idx = shuffle // num_cols
        col_idx = shuffle % num_cols
        ax = axes[row_idx, col_idx]
        ax.plot(np.arange(1, len(role_AB) + 1), role_AB, '-*r', lw=10, markersize=20)
        ax.plot(np.arange(1, len(role_BA) + 1), role_BA, '-*b', lw=10, markersize=20)
        ax.set_xlabel('Iterations', fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel('Relative Error', fontsize=16)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                   markerfacecolor='blue', markersize=20),
                           Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                   markerfacecolor='red', markersize=20)]

        ax.legend(handles=legend_elements, title='{:}'.format(thee), title_fontsize='x-large', fontsize=16)
        ax.set_title('Shuffle = {:}'.format(shuffle+1), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



from sklearn.metrics import confusion_matrix,balanced_accuracy_score
# if datum in ['depth_results_immature_20230706']:
#         lambd_values = np.linspace(0.1,0.5,10)
#         num_iterations_values = [5,10,15,20,25,30,35,40,50,100,200,250,300,500]  
#         beta_values = np.linspace(0.1,100,10)
#         if reg in ['lq1','lq12']:
#             q = kwargs['q']
#             epsa = kwargs['epsa']

[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2]
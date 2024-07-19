import numpy as np
from kneefinder import KneeFinder
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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

def combined_plots(X_train, X_test, y_train, y_test, dic_pgd, pp, pp1, pp2, pp3, pp4,thee,index,reg,sep,**kwargs):

    A = X_train
    B = X_test

    qa = len(y_train[y_train == 0])
    qb = len(y_test[y_test == 0])

    if reg == 'lq1':
        q = kwargs['q']

    if sep == 'npca':
        fig, ax = plt.subplots(1, 3, figsize=(28, 8))
        fig.suptitle(pp, fontsize=20)

        Apgd = dic_pgd['c']
        Bpgd = dic_pgd['d']
        
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
            ax[0].plot(w_A, '-*b', ms=10)
        else:
            ax[0].plot(w_A, '--*b', ms=20, lw=5)
        ax[0].set_ylabel('sorted weights', fontsize=16)
        ax[0].set_xlabel('sorted weight index', fontsize=16)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[0].plot(xp, yp, "or", ms=10)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                    markerfacecolor='blue', markersize=20),
                            Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                    markerfacecolor='red', markersize=20)]
        if reg == 'l1':
            ax[0].legend(handles=legend_elements, title=f'{thee}', title_fontsize='x-large', fontsize=16)
        elif reg == 'lq1':
            ax[0].legend(handles=legend_elements, title=f'{thee}, $q=${q}', title_fontsize='x-large', fontsize=16)
        for a, b in zip([int(xp)], [round(yp, 3)]):
            ax[0].annotate(f'({a+1}, {b})', xy=(a, b), xytext=(10, 10), fontsize=15, textcoords='offset points', arrowprops=dict())
        if m1 < n1:
            ax[0].plot(w_B,'-*r', ms=10)
        else:
            ax[0].plot(w_B, '--*r', ms=20, lw=5)
        ax[0].set_ylabel('Sorted Weights', fontsize=16)
        ax[0].set_xlabel('Sorted Weight Index', fontsize=16)
        ax[0].plot(xm, ym, "ob", ms=10)
        for a, b in zip([int(xm)], [round(ym, 3)]):
            ax[0].annotate(f'({a+1}, {b})', xy=(a, b), xytext=(30, 100), fontsize=15, textcoords='offset points', arrowprops=dict())
    
        

        role_AB = dic_pgd['a']
        role_BA = dic_pgd['b']
        ax[1].plot(np.arange(1, len(role_AB) + 1), role_AB, '-*r', lw=10, markersize=20)
        ax[1].plot(np.arange(1, len(role_BA) + 1), role_BA, '-*b', lw=10, markersize=20)
        ax[1].set_xlabel('Iterations', fontsize=16)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].set_ylabel('Objective Function', fontsize=16)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                    markerfacecolor='blue', markersize=20),
                            Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                    markerfacecolor='red', markersize=20)]
        if reg == 'l1':
            ax[1].legend(handles=legend_elements, title=f'{thee}', title_fontsize='x-large', fontsize=16)
        elif reg == 'lq1':
            ax[1].legend(handles=legend_elements, title=f'{thee}, $q=${q}', title_fontsize='x-large', fontsize=16)
        

        role_AB = dic_pgd['e']
        role_BA = dic_pgd['f']
        ax[2].plot(np.arange(1, len(role_AB) + 1), role_AB, '-*r', lw=10, markersize=20)
        ax[2].plot(np.arange(1, len(role_BA) + 1), role_BA, '-*b', lw=10, markersize=20)
        ax[2].set_xlabel('Iterations', fontsize=16)
        ax[2].tick_params(axis='x', labelsize=16)
        ax[2].tick_params(axis='y', labelsize=16)
        ax[2].set_ylabel('Relative Error', fontsize=16)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                    markerfacecolor='blue', markersize=20),
                            Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                    markerfacecolor='red', markersize=20)]
        if reg == 'l1':
            ax[2].legend(handles=legend_elements, title=f'{thee}', title_fontsize='x-large', fontsize=16)
        elif reg == 'lq1':
            ax[2].legend(handles=legend_elements, title=f'{thee}, $q=${q}', title_fontsize='x-large', fontsize=16)

        # fid_A = np.log(dic_pgd['g'])
        # reg_A = np.log(dic_pgd['h'])
        # fid_B = np.log(dic_pgd['i'])
        # reg_B = np.log(dic_pgd['j'])
        # plt.plot(reg_A,fid_A, label='x', color='blue')
        # plt.plot(reg_B,fid_B, label='y', color='blue', linestyle='--')
        # plt.xlabel('Iterations')
        # plt.ylabel('Values')
        # plt.legend()

        plt.tight_layout() #rect=[0, 0, 1, 0.96]
        plt.show()


    elif sep == 'pca':
        fig, ax = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(pp, fontsize=20)

        Apgd = dic_pgd['c']
        Bpgd = dic_pgd['d']
        
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
            ax[0].plot(w_A, '-*b', ms=10)
        else:
            ax[0].plot(w_A, '--*b', ms=20, lw=5)
        ax[0].set_ylabel('sorted weights', fontsize=16)
        ax[0].set_xlabel('sorted weight index', fontsize=16)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[0].plot(xp, yp, "or", ms=10)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                    markerfacecolor='blue', markersize=20),
                            Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                    markerfacecolor='red', markersize=20)]
        if reg == 'l1':
            ax[0].legend(handles=legend_elements, title=f'{thee}', title_fontsize='x-large', fontsize=16)
        elif reg == 'lq1':
            ax[0].legend(handles=legend_elements, title=f'{thee}, $q=${q}', title_fontsize='x-large', fontsize=16)
        for a, b in zip([int(xp)], [round(yp, 3)]):
            ax[0].annotate(f'({a+1}, {b})', xy=(a, b), xytext=(10, 10), fontsize=15, textcoords='offset points', arrowprops=dict())
        if m1 < n1:
            ax[0].plot(w_B,'-*r', ms=10)
        else:
            ax[0].plot(w_B, '--*r', ms=20, lw=5)
        ax[0].set_ylabel('Sorted Weights', fontsize=16)
        ax[0].set_xlabel('Sorted Weight Index', fontsize=16)
        ax[0].plot(xm, ym, "ob", ms=10)
        for a, b in zip([int(xm)], [round(ym, 3)]):
            ax[0].annotate(f'({a+1}, {b})', xy=(a, b), xytext=(30, 100), fontsize=15, textcoords='offset points', arrowprops=dict())
    
        

        role_AB = dic_pgd['a']
        role_BA = dic_pgd['b']
        ax[1].plot(np.arange(1, len(role_AB) + 1), role_AB, '-*r', lw=10, markersize=20)
        ax[1].plot(np.arange(1, len(role_BA) + 1), role_BA, '-*b', lw=10, markersize=20)
        ax[1].set_xlabel('Iterations', fontsize=16)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].set_ylabel('Objective Function', fontsize=16)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                    markerfacecolor='blue', markersize=20),
                            Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                    markerfacecolor='red', markersize=20)]
        if reg == 'l1':
            ax[1].legend(handles=legend_elements, title=f'{thee}', title_fontsize='x-large', fontsize=16)
        elif reg == 'lq1':
            ax[1].legend(handles=legend_elements, title=f'{thee}, $q=${q}', title_fontsize='x-large', fontsize=16)
        

        role_AB = dic_pgd['e']
        role_BA = dic_pgd['f']
        ax[2].plot(np.arange(1, len(role_AB) + 1), role_AB, '-*r', lw=10, markersize=20)
        ax[2].plot(np.arange(1, len(role_BA) + 1), role_BA, '-*b', lw=10, markersize=20)
        ax[2].set_xlabel('Iterations', fontsize=16)
        ax[2].tick_params(axis='x', labelsize=16)
        ax[2].tick_params(axis='y', labelsize=16)
        ax[2].set_ylabel('Relative Error', fontsize=16)
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp1),
                                    markerfacecolor='blue', markersize=20),
                            Line2D([0], [0], marker='*', color='w', label='{:}'.format(pp2),
                                    markerfacecolor='red', markersize=20)]
        if reg == 'l1':
            ax[2].legend(handles=legend_elements, title=f'{thee}', title_fontsize='x-large', fontsize=16)
        elif reg == 'lq1':
            ax[2].legend(handles=legend_elements, title=f'{thee}, $q=${q}', title_fontsize='x-large', fontsize=16)

        # fid_A = np.log(dic_pgd['g'])
        # reg_A = np.log(dic_pgd['h'])
        # fid_B = np.log(dic_pgd['i'])
        # reg_B = np.log(dic_pgd['j'])
        # plt.plot(reg_A,fid_A, label='x', color='blue')
        # plt.plot(reg_B,fid_B, label='y', color='blue', linestyle='--')
        # plt.xlabel('Iterations')
        # plt.ylabel('Values')
        # plt.legend()


        CC = np.concatenate((A[0:qa, :], A[qa:A.shape[0], :]), axis=0)
        CD = np.concatenate((B[0:qb, :], B[qb:B.shape[0], :]), axis=0)
        EF = np.concatenate((CC, CD),axis=0)
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        DC = scaler.fit_transform(EF)
        X_pca = pca.fit_transform(DC)
        labs = np.array([0] * qa  + [1] * (A.shape[0] - qa) + [2]*qb + [3] * (B.shape[0] - qb))
        ax[3].scatter(X_pca[labs == 0, 0], X_pca[labs == 0, 1], color='blue', label='{:}'.format(pp1), marker='o', s=200)
        ax[3].scatter(X_pca[labs == 1, 0], X_pca[labs == 1, 1], color='red', label='{:}'.format(pp2), marker='o', s=200)
        ax[3].scatter(X_pca[labs == 2, 0], X_pca[labs == 2, 1], color='g', label='{:}'.format(pp3), marker='X', s=200)
        ax[3].scatter(X_pca[labs == 3, 0], X_pca[labs == 3, 1], color='k', label='{:}'.format(pp4), marker='X', s=200)
        ax[3].legend(title='{:}'.format(thee), title_fontsize='x-large', fontsize=16)
        ax[3].set_xlabel('PC1', fontsize=16)
        ax[3].set_ylabel('PC2', fontsize=16)
        ax[3].tick_params(axis='x', labelsize=16)
        ax[3].tick_params(axis='y', labelsize=16)

        # a_w = np.abs(Apgd[0:A.shape[1]]).argsort()[::-1]
        # b_w = np.abs(Bpgd[0:A.shape[1]]).argsort()[::-1]
        # rho = min(myelbow(Apgd[:-1],Bpgd[:-1]))
        # indexx = int(rho) + 1
        # CC = np.concatenate((A[0:qa, index], A[qa:A.shape[0], index]), axis=0)
        # CD = np.concatenate((B[0:qb, index], B[qb:B.shape[0], index]), axis=0)
        
        EF = np.concatenate((A[:,index], B[:,index]),axis=0)
        pca = PCA(n_components=2)
        DC = StandardScaler().fit_transform(EF)
        X_pca = pca.fit_transform(DC)
        labs = np.array([0] * qa  + [1] * (A.shape[0] - qa) + [2]*qb + [3] * (B.shape[0] - qb))
        ax[4].scatter(X_pca[labs == 0, 0], X_pca[labs == 0, 1], color='blue', label='{:}'.format(pp1), marker='o', s=200)
        ax[4].scatter(X_pca[labs == 1, 0], X_pca[labs == 1, 1], color='red', label='{:}'.format(pp2), marker='o', s=200)
        ax[4].scatter(X_pca[labs == 2, 0], X_pca[labs == 2, 1], color='g', label='{:}'.format(pp3), marker='X', s=200)
        ax[4].scatter(X_pca[labs == 3, 0], X_pca[labs == 3, 1], color='k', label='{:}'.format(pp4), marker='X', s=200)
        ax[4].legend(title='{:}'.format(thee), title_fontsize='x-large', fontsize=16)
        ax[4].set_xlabel('PC1', fontsize=16)
        ax[4].set_ylabel('PC2', fontsize=16)
        ax[4].tick_params(axis='x', labelsize=16)
        ax[4].tick_params(axis='y', labelsize=16)

        plt.tight_layout() #rect=[0, 0, 1, 0.96]
        plt.show()



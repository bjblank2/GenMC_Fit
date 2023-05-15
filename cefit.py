import symop
import copy
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import resample

n = 1000
alpha_range = [-7, 2]
alpha_cv = np.logspace(alpha_range[0], alpha_range[1], num=100)
l1 = [.4, .5, .6, .7, .9]
kf = KFold(n_splits=10, shuffle=True, random_state=123)


def ridge_fit(count, enrg):
    """
    Lasso fitting to raw energy per atom
    :param enrg: energy list
    :param count: count list containing clusters, decorations, and counts
    :return: list of ECIs
    """
    # bootstrap ridge
    print('Ridge: alpha, rmse, score, coef_num', flush=True)
    attr_list = [[] for _ in range(n)]
    coef_list = [[] for _ in range(n)]
    for i in range(n):
        x, y = resample(count, enrg, n_samples=250, random_state=i)
        model = RidgeCV(alphas=alpha_cv, cv=kf)
        model.fit(x, y)
        coef_list[i].append(model.intercept_)
        coef_list[i].extend(model.coef_.tolist())
        rmse = np.sqrt(mean_squared_error(model.predict(count), enrg))
        score = model.score(count, enrg)
        coef_num = np.sum(model.coef_ != 0)
        print(model.alpha_, rmse, score, coef_num, flush=True)
        attr_list[i].extend([model.alpha_, rmse, score, float(coef_num)])
    coef_mean = np.mean(coef_list, axis=0)
    with open('ridge_coef', 'w') as filehandle:
        json.dump(coef_list, filehandle)
    with open('ridge_attr', 'w') as filehandle:
        json.dump(attr_list, filehandle)

    return coef_mean


def lasso_fit(count, enrg):
    """
    Lasso fitting to energy above hull
    :param enrg: energy list
    :param count: count list containing clusters, decorations, and counts
    :return: list of ECIs
    """
    # bootstrap lasso
    print('alpha, rmse, score, coef_num', flush=True)
    coef_list = [[] for _ in range(n)]
    attr_list = [[] for _ in range(n)]
    for i in range(n):
        x, y = resample(count, enrg, n_samples=250, random_state=i)
        model = LassoCV(alphas=alpha_cv, cv=kf, max_iter=10000000, tol=1e-5)
        model.fit(x, y)
        coef_list[i].append(model.intercept_)
        coef_list[i].extend(model.coef_.tolist())
        rmse = np.sqrt(mean_squared_error(model.predict(count), enrg))
        score = model.score(count, enrg)
        coef_num = np.sum(model.coef_ != 0)
        print(model.alpha_, rmse, score, coef_num, flush=True)
        attr_list[i].extend([model.alpha_, rmse, score, float(coef_num)])
    coef_mean = np.mean(coef_list, axis=0)
    print('# of lasso selected features', np.sum(coef_mean != 0), flush=True)
    with open('lasso_coef', 'w') as filehandle:
        json.dump(coef_list, filehandle)
    with open('lasso_attr', 'w') as filehandle:
        json.dump(attr_list, filehandle)

    return coef_mean


def eln_fit(count, enrg):
    """
    ElasticNet fitting to raw energy per atom
    :param enrg: energy list
    :param count: count list containing clusters, decorations, and counts
    :return: list of ECIs
    """
    # bootstrap elasticnet
    print('Eln: alpha, l1_ratio, rmse, score, coef_num', flush=True)
    attr_list = [[] for _ in range(n)]
    coef_list = [[] for _ in range(n)]
    for i in range(n):
        x, y = resample(count, enrg, n_samples=250, random_state=i)
        model = ElasticNetCV(alphas=alpha_cv, cv=kf, max_iter=10000000, tol=1e-5, l1_ratio=l1)
        model.fit(x, y)
        coef_list[i].append(model.intercept_)
        coef_list[i].extend(model.coef_.tolist())
        rmse = np.sqrt(mean_squared_error(model.predict(count), enrg))
        score = model.score(count, enrg)
        coef_num = np.sum(model.coef_ != 0)
        print(model.alpha_, model.l1_ratio_, rmse, score, coef_num, flush=True)
        attr_list[i].extend([model.alpha_, model.l1_ratio_, rmse, score, float(coef_num)])
    coef_mean = np.mean(coef_list, axis=0)
    print('# of eln selected features', np.sum(coef_mean != 0), flush=True)
    with open('eln_coef', 'w') as filehandle:
        json.dump(coef_list, filehandle)
    with open('eln_attr', 'w') as filehandle:
        json.dump(attr_list, filehandle)

    return coef_mean


def write_eci(symeq_clust_list, deco_list, eci_list, pntsym_list, spec_seq):
    """
    write the clusters and ecis as a rule file for the magnetic MC simulation
    :param symeq_clust_list:
    :param deco_list:
    :param eci_list:
    :param pntsym_list:
    :param spec_seq
    :return: a file like this
            #
            Motif= 0, 0, 0 : 1, 0, 0 : 0, 1, 0
            Deco= 0, 0, 0 : 1, 1, 1 : 0, 1, 1 : 1, 0, 0 : 2, 2, 1
            Type= 0, 0, 0, 0, 0
            Enrg = -0.002, 0.01, -0.025, -0.012, 1.1
            #
            Motif= 0, 0, 0 : 0, 1, 0
            Deco= 0, 0 : 1, 1 : 0, 1 : 2, 1
            Type= 1, 1, 0, 0
            Enrg = -0.003, 0.02, -0.02, -0.01
    """
    output = open('MC_rules', 'w')
    output.write('# \n')
    output.write('Motif : intercept \n')
    output.write('Enrg : ' + str(eci_list[0]) + '\n')
    for i in range(len(symeq_clust_list)):
        start = 0
        for k in range(0, i):
            start = start + len(deco_list[k])
        for j in range(len(symeq_clust_list[i])):
            clust = symeq_clust_list[i][j]
            motif = clust[0]
            deco = deco_list[i]
            spin = symeq_clust_list[i][j][2][0]
            enrg_list = []
            output.write('# \n')
            output.write('Motif')
            for k in range(len(motif)):
                output.write(' : ' + ', '.join(map(str, motif[k])) )
            output.write('\nDeco')
            for k in range(len(deco)):
                spec_list = symop.find_eq_spec_list(deco[k], clust, pntsym_list[i][j], spec_seq)
                enrg = [eci_list[start + k + 1]] * len(spec_list)
                enrg_list.extend(enrg)
                for m in range(len(spec_list)):
                    output.write(' : ' + str(spec_list[m]))
            output.write('\nType : ' + str(spin))
            output.write('\nEnrg')
            for k in range(len(enrg_list)):
                output.write(' : ' + str(enrg_list[k]))
            output.write('\n')
    output.close()

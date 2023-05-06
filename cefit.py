import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def ridge_fit(count, enrg):
    """
    Lasso fitting to raw energy per atom
    :param enrg: energy list
    :param count: count list containing clusters, decoration, and counts
    :return: list of ECIs
    """
    # train-test ridge
    x_train, x_test, y_train, y_test = train_test_split(count, enrg, test_size=0.2, random_state=184)
    alpha_range = [-6, 3]
    alpha_train = np.logspace(alpha_range[0], alpha_range[1], num=50)
    train_rmse = []
    test_rmse = []
    train_score = []
    test_score = []
    coeff_num = []
    for a in alpha_train:
        model = Ridge(alpha=a).fit(x_train, y_train)
        train_rmse.append(np.sqrt(mean_squared_error(model.predict(x_train), y_train)))
        test_rmse.append(np.sqrt(mean_squared_error(model.predict(x_test), y_test)))
        train_score.append(model.score(x_train, y_train))
        test_score.append(model.score(x_test, y_test))
        coeff_num.append(np.sum(model.coef_ != 0))
    for i in range(len(test_rmse)):
        if test_rmse[i] == np.min(test_rmse):
            print('ridge', test_rmse[i], test_score[i])
            print(alpha_train[i], np.log10(alpha_train[i]))
            print(coeff_num[i])
    plt.ylim(0, 0.05)
    plt.scatter(np.log10(alpha_train), train_rmse, label='train rmse')
    plt.scatter(np.log10(alpha_train), test_rmse, label='test rmse')
    plt.legend()
    plt.savefig('ridge_train.pdf')

    # bootstrap ridge
    ridge_rmse_list = [[] for _ in range(100)]
    ridge_score_list = [[] for _ in range(100)]
    ridge_coef_num = [[] for _ in range(100)]
    ridge_coef_list = [[] for _ in range(100)]
    ridge_alpha_list = [[] for _ in range(100)]
    ridge_model_list = [[] for _ in range(100)]
    for i in range(100):
        x, y = resample(count, enrg, n_samples=150, random_state=i)
        alpha_range = [-4, 1]
        alpha_ridge = np.logspace(alpha_range[0], alpha_range[1], num=50)
        kf = KFold(n_splits=10, shuffle=True)
        model = RidgeCV(alphas=alpha_ridge, cv=kf).fit(x, y)
        ridge_model_list[i] = model
        ridge_rmse_list[i].append(np.sqrt(mean_squared_error(model.predict(x), y)))
        ridge_score_list[i].append(model.score(x, y))
        ridge_coef_num[i].append(np.sum(model.coef_ != 0))
        ridge_coef_list[i].append(model.intercept_)
        ridge_coef_list[i].extend(model.coef_.tolist())
        ridge_alpha_list[i].append(model.alpha_)
    coef_mean = np.mean(ridge_coef_list, axis=0)
    coef_err = np.std(ridge_coef_list, axis=0)
    plt.figure(dpi=300, figsize=(12, 3))
    plt.errorbar(np.arange(len(coef_mean)), coef_mean, yerr=coef_err, fmt='ro', ms=3, ecolor='g', capsize=2)
    plt.savefig('ridge_bootstrap.pdf')

    return coef_mean


def lasso_fit(count, enrg):
    """
    Lasso fitting to energy above hull
    :param enrg: energy list
    :param count: count list containing clusters, decoration, and counts
    :return: list of ECIs
    """
    # train-test lasso
    x_train, x_test, y_train, y_test = train_test_split(count, enrg, test_size=0.2, random_state=237)
    alpha_range = [-6, -1]
    alpha_train = np.logspace(alpha_range[0], alpha_range[1], num=50)
    train_rmse = []
    test_rmse = []
    train_score = []
    test_score = []
    coeff_num = []
    for a in alpha_train:
        model = Lasso(alpha=a, max_iter=10000000, tol=1e-6).fit(x_train, y_train)
        train_rmse.append(np.sqrt(mean_squared_error(model.predict(x_train), y_train)))
        test_rmse.append(np.sqrt(mean_squared_error(model.predict(x_test), y_test)))
        train_score.append(model.score(x_train, y_train))
        test_score.append(model.score(x_test, y_test))
        coeff_num.append(np.sum(model.coef_ != 0))
    for i in range(len(test_rmse)):
        if test_rmse[i] == np.min(test_rmse):
            print(test_rmse[i], test_score[i])
            print(alpha_train[i], np.log10(alpha_train[i]))
            print(coeff_num[i])
    plt.ylim(0, 0.05)
    plt.scatter(np.log10(alpha_train), train_rmse, label='train rmse')
    plt.scatter(np.log10(alpha_train), test_rmse, label='test rmse')
    plt.legend()
    plt.savefig('lasso_train.pdf')

    # bootstrap lasso
    lasso_rmse_list = [[] for _ in range(100)]
    lasso_score_list = [[] for _ in range(100)]
    lasso_coef_num = [[] for _ in range(100)]
    lasso_coef_list = [[] for _ in range(100)]
    lasso_alpha_list = [[] for _ in range(100)]
    lasso_model_list = [[] for _ in range(100)]
    for i in range(100):
        x, y = resample(count, enrg, n_samples=150, random_state=i)
        alpha_range = [-6, -1]
        alpha_lasso = np.logspace(alpha_range[0], alpha_range[1], num=50)
        kf = KFold(n_splits=10, shuffle=True)
        model = LassoCV(alphas=alpha_lasso, cv=kf, max_iter=10000000, tol=1e-5).fit(x, y)
        lasso_model_list[i] = model
        lasso_rmse_list[i].append(np.sqrt(mean_squared_error(model.predict(x), y)))
        lasso_score_list[i].append(model.score(x, y))
        lasso_coef_num[i].append(np.sum(model.coef_ != 0))
        lasso_coef_list[i].append(model.intercept_)
        lasso_coef_list[i].extend(model.coef_.tolist())
        lasso_alpha_list[i].append(model.alpha_)
    coef_mean = np.mean(lasso_coef_list, axis=0)
    coef_err = np.std(lasso_coef_list, axis=0)
    plt.figure(dpi=300, figsize=(12, 3))
    plt.errorbar(np.arange(len(coef_mean)), coef_mean, yerr=coef_err, fmt='ro', ms=3, ecolor='g', capsize=2)
    plt.savefig('lasso_bootstrap.pdf')

    return coef_mean













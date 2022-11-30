import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm_notebook
from src.data import get_market_index
from scipy import stats
from sklearn import decomposition


def get_return_mean_cov(stocks):
    # получить по выбранным активам матрицу их доходностей,
    # вектор средних доходностей и матрицу ковариации
    r_matrix = {}
    min_data_len = len(min(stocks, key=lambda x: len(x.profitability[0])).profitability[0])
    for stock in stocks:
        r_matrix[stock.key] = stock.profitability[0].copy()[:min_data_len - 1]
    r_df = pd.DataFrame(r_matrix).dropna()
    return r_df.values, r_df.mean().values, r_df.cov().values


# Оценка портфелей для которыйх короткие продажи разрешены
def risk_function_for_portfolio(X, cov_matrix, n_observations=1, sqrt=True):
    # оценка риска портфеля
    if sqrt:
        return np.sqrt(np.dot(np.dot(X, cov_matrix), X.T))
    else:
        return np.dot(np.dot(X, cov_matrix), X.T) / np.sqrt(n_observations)


def optimize_portfolio(risk_estimation_function,
                       returns,
                       mean_returns,
                       cov_matrix,
                       bounds,
                       target_return=None):
    # оптимизатор с итеративным методом МНК SLSQP решает задачу мимнимизации уравнения Лагранжа
    X = np.ones(returns.shape[1])
    X = X / X.sum()
    bounds = bounds * returns.shape[1]

    constraints = [{'type': 'eq', 'fun': lambda X_: np.sum(X_) - 1.0}]
    if target_return:
        constraints.append({'type': 'eq',
                            'args': (mean_returns,),
                            'fun': lambda X_, mean_returns_: target_return - np.dot(X_, mean_returns_)})

    return minimize(risk_estimation_function, X,
                    args=(cov_matrix, returns.shape[0]),
                    method='SLSQP',
                    constraints=constraints,
                    bounds=bounds).x


# эту функцию мы должны минимизировать
def objective_function(X, returns, gamma, cov_matrix):
    # gamma - наше отношение к риску
    return - np.dot(returns, X) + gamma * risk_function_for_portfolio(X, cov_matrix)


def objective_function_for_model(x, cov_matrix, mean_vector, risk_free_mean):
    return float(-(x.dot(mean_vector) - risk_free_mean) / np.sqrt(np.dot(np.dot(x, cov_matrix), x.T)))


# Ищем оптимальный портфель, решаем задачу оптимизации
def search_optimal_portfolio_with_attitude_to_risk(selected_objective_function, returns, cov_matrix, gamma, bounds, N):
    X = np.ones(N)
    X = X / X.sum()
    bounds = bounds * N

    # линейное ограничение, что сумма долей должна быть равна 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    return minimize(selected_objective_function, X, args=(returns, gamma, cov_matrix), method='SLSQP',
                    constraints=constraints, bounds=bounds).x


def risk_aversion_computing(stocks, short_selling_is_allowed, gammas):
    risk_of_the_optimal_portfolio_with_minimal_risk = []
    profitability_of_the_optimal_portfolio_with_minimal_risk = []
    losses = {}
    N = len(stocks)  # количество активов
    E = []
    for stock in stocks:
        E.append(stock.E)

    # Возвращает много чего интересного матрицу, вектор, матрицу ковариации.
    r_matrix, mean_vec, cov_matrix = get_return_mean_cov(stocks)
    if short_selling_is_allowed:
        bounds = ((-1, 1),)
    else:
        bounds = ((0, 1),)
    for gamma in gammas:
        shares_of_the_optimal_portfolio_with_minimal_risk = search_optimal_portfolio_with_attitude_to_risk(
            objective_function, E, cov_matrix, gamma, bounds, N)
        risk_of_the_optimal_portfolio_with_minimal_risk.append(
            risk_function_for_portfolio(shares_of_the_optimal_portfolio_with_minimal_risk, cov_matrix))
        profitability_of_the_optimal_portfolio_with_minimal_risk.append(
            np.dot(shares_of_the_optimal_portfolio_with_minimal_risk, E))
        losses[gamma] = - np.dot(r_matrix, shares_of_the_optimal_portfolio_with_minimal_risk)
    return risk_of_the_optimal_portfolio_with_minimal_risk, profitability_of_the_optimal_portfolio_with_minimal_risk, \
           losses


def VaR_for_portfolios(gammas, losses_):
    confidence_lvl = 0.95
    VaR = {}
    for gamma in gammas:
        print('VaR with confidence level %s:' % gamma)
        loss = losses_[gamma]
        loss = loss[np.isfinite(loss)]
        VaR[confidence_lvl] = np.quantile(loss, confidence_lvl)
        print('Losses will not exceed %.4f with %.2f%s certainty.' % (
            np.round(VaR[confidence_lvl], 4), confidence_lvl, '%'))


def optimal_portfolio_sharp_ratio(virtual_stock_E, N, cov_matrix, E, bounds):
    X = np.ones(N)
    X = X / X.sum()
    bounds = bounds * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    minimized = minimize(objective_function_for_model,
                         X,
                         args=(cov_matrix, E, virtual_stock_E,),
                         method='SLSQP',
                         constraints=constraints,
                         bounds=bounds).x
    return minimized


def optimal_portfolio_computing(stocks, virtual_stock, short_is_allowed):
    virtual_stock_E = virtual_stock[1]  ## E
    N = len(stocks)
    r_matrix, _, cov_matrix = get_return_mean_cov(stocks)
    E = []
    for stock in stocks:
        E.append(stock.E)

    if short_is_allowed:
        bounds = ((-1, 1),)
    else:
        bounds = ((0, 1),)

    optimum_portfolio_weights = optimal_portfolio_sharp_ratio(virtual_stock_E, N, cov_matrix, E, bounds)

    the_best_risk_sharp = risk_function_for_portfolio(optimum_portfolio_weights, cov_matrix)
    the_best_E_sharp = np.dot(optimum_portfolio_weights, E)
    losses = -np.dot(r_matrix, optimum_portfolio_weights)

    return the_best_risk_sharp, the_best_E_sharp, losses


def VaR_for_portfolio(losses):
    confidence_lvl = 0.95
    loss = losses[np.isfinite(losses)]
    print(
        'Losses will not exceed %.4f with %.2f%s certainty.' % (np.quantile(loss, confidence_lvl), confidence_lvl, '%'))


def count_virtual_stock_without_risk(stocks):
    stocks_sorted_risks = sorted(stocks, key=lambda x: x.risk)
    sum_el = 3
    average_E = 0.0
    for stock in stocks_sorted_risks[:sum_el - 1]:
        average_E += stock.E
    average_E /= sum_el
    stocks_sorted_E = sorted(stocks_sorted_risks[:sum_el - 1], key=lambda x: x.E)
    downgrade_to = abs(stocks_sorted_E[0].E - stocks_sorted_E[-1].E)/2.0
    return [0, average_E - downgrade_to]

def VaR_info(losses):
    confidence_lvl = [0.9, 0.95, 0.99]
    VaR = {}
    for clvl in confidence_lvl:
        loss = losses[np.isfinite(losses)]
        VaR[clvl] = np.quantile(loss, clvl)
        print(' - Потери не превысят %.4f с %.2f%s уверенностью.' % (VaR[clvl], clvl, '%'))


def pca_algorithm(painter, stocks, df_for_graph, n_components, index):
    pca = decomposition.PCA(n_components=n_components, random_state=12)
    r_matrix, mean_vec, cov_matrix = get_return_mean_cov(stocks)
    df = pd.DataFrame(r_matrix)
    df = df.dropna()
    r_matrix = df.values
    pca.fit(r_matrix)
    pca_matrix = pca.transform(r_matrix)
    painter.plot_pca(pca_matrix)
    painter.plot()
    print(str(n_components) + " component significance: " + str(sum(pca.explained_variance_ratio_[:n_components])))
    i = 0
    weights = pca.components_
    Xpca = weights[i] / sum(weights[i])
    print('Xpca sum = ' + str(Xpca.sum()))
    VaR_info(index.profitability)
    for i in range(n_components):
        Xpca = weights[i] / sum(weights[i])
        Xpca.sum()
        losses = -np.dot(r_matrix, Xpca)
        print('Component ', i + 1)
        VaR_info(losses)

    painter.plot_stock_map(df_for_graph, "6")
    painter.plot_point(index.risk, index.E, 'yellow', '*', 'BOVESPA индекс рынка')
    pca_index_risk = risk_function_for_portfolio(Xpca, cov_matrix)
    pca_index_return = np.dot(Xpca, mean_vec)
    painter.plot_point(pca_index_risk, pca_index_return, 'black', '*', 'PCA индекс рынка')
    painter.plot()


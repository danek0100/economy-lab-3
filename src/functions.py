import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm_notebook
from data import get_market_index
from scipy import stats
from sklearn import decomposition
from constant_strings import *
from cvxopt import matrix, solvers
from IPython.display import clear_output
from scipy.stats import norm
import pprint

def get_parameter_b(beta):
    return (np.sqrt(2 * np.pi)*(1 - beta) )**(-1) * np.exp(-(norm.cdf(norm.ppf(beta))**2 / 2)) 

def get_return_mean_cov(stocks):
    # получить по выбранным активам матрицу их доходностей,
    # вектор средних доходностей и матрицу ковариации
    r_matrix = {}
    min_data_len = len(min(stocks, key=lambda x: len(x.profitability[0])).profitability[0])
    for stock in stocks:
        r_matrix[stock.key] = stock.profitability[0].copy()[:min_data_len - 1]
    r_df = pd.DataFrame(r_matrix).dropna()
    return r_df.values, r_df.mean().values, r_df.cov().values

def risk_portfolio(X, cov_matrix):
    return np.sqrt(np.dot(np.dot(X, cov_matrix), X.T))
    
def objective_function(X, mean_vec, cov_matrix, b):
    return (-np.dot(mean_vec, X)) + b * np.dot(np.dot(X, cov_matrix), X.T)
    

def optimize_portfolio(mean_vec,cov_matrix, b, bounds, objective_function=objective_function, cvxopt=False):
    N = cov_matrix.shape[0]
    X = np.ones(N)
    X = X / X.sum()
    bounds = bounds * N
    constraints=[]
    constraints.append({'type': 'eq', 'fun': lambda X: np.sum(X) - 1.0})
    result = minimize(objective_function, X, args=(mean_vec, cov_matrix, b), method='SLSQP', constraints=constraints, bounds=bounds).x
    return result
    
def cvar_objective_function(UXalpha, T, beta):
    return UXalpha[-1] + 1 / (T * (1 - beta)) * np.sum(UXalpha[:T])
    
def cvar_optimize_portfolio(generated_data, beta, cvar_objective_function_ =cvar_objective_function):
    alpha  = 0 
    N = generated_data.shape[1]
    X = np.ones(N)/ N 
   
    T = generated_data.shape[0] 
    U = np.dot(generated_data,  X) - alpha
    
    UXalpha = np.zeros(T+N+1)
    UXalpha[:T] = U
    UXalpha[T:N+T]= X
    UXalpha[-1] = alpha
    
    bounds_U = ((0, 100000000000),) * T
    bounds_X = ((0, 1.1),) * N
    bounds_alpha = ((-100000, 100000),)
    bounds = bounds_U + bounds_X + bounds_alpha
    
    
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda X: sum(X[T:N+T]) -1})
    def u_x_con(UXalpha, generated_data, i):
        return np.dot(generated_data[i], UXalpha[T:N+T]) + UXalpha[-1] - UXalpha[i],
    for i in range(T):
        constraints.append({'type': 'ineq', 'fun': u_x_con,'args': (generated_data, i)})

    return minimize(cvar_objective_function_, UXalpha,
                    args=(T, beta), method='SLSQP',
                    constraints=constraints,
                    bounds=bounds).x
    
    

def plot_shares(shares, stocks):
    plt.grid()
    data = pd.DataFrame()
    i = 0
    for stock in stocks:
        data.insert(0 , str(stock.key) , [shares[i]])
        i += 1
    plt.title('Shares of stocks within the portfolio', size = 20)
    plt.ylabel('Shares', size = 15)
    plt.xlabel('Stocks', size = 15)
    sns.barplot(data)
    plt.show()

def plot_map_with_one_portfolio(X, mean_vec, cov_matrix, all_stocks, b):
    plt.grid()
    plt.scatter(data=all_stocks, x='σ', y='E', c='#6C8CD5', label='Stocks', edgecolors = 'white')
    plt.legend(["Stocks"])
    plt.scatter(risk_portfolio(X, cov_matrix), np.dot(mean_vec,X), color = 'red', s = 200, marker='^', edgecolors='white', label='Optimal portfolio with b = %.2f' % b)
    plt.xlabel('σ', size=15)
    plt.ylabel('E', size=15)
    plt.title('Profitability/Risk Map', size=20)
    plt.legend()
    plt.show()

def plot_map_with_two_portfolio(all_stocks, cov_matrix, mean_vec, X, X_est):
    plt.grid()
    plt.scatter(data=all_stocks, x='σ', y='E', c='#6C8CD5', label='Stocks')
    plt.legend(["Stocks"])
    plt.scatter(risk_portfolio(X, cov_matrix), np.dot(mean_vec,X), color = 'red', s = 200, marker='^', edgecolors='white', label='True portfolio')
    plt.scatter(risk_portfolio(X_est, cov_matrix), np.dot(mean_vec,X_est), color ='green', s = 200, marker='^', edgecolors='white', label='Sample portfolio' )
    plt.xlabel('σ', size=15)
    plt.ylabel('E', size=15)
    plt.title('Profitability/Risk Map', size=16)
    plt.legend()
    plt.show()


def plot_result_experiments(df_for_graph_selected, experiments, shares, cov_matrix, mean_vec, b ):
    plt.grid()
    plt.scatter(data=df_for_graph_selected, x='σ', y='E', c='#6C8CD5', label='Stocks', edgecolors='white')
    plt.scatter(risk_portfolio(shares, cov_matrix),
            np.dot(mean_vec,shares), 
            c='red',
            marker='^',
            s=300, 
            edgecolors='white',
            label='Optimal portfolio with b = %.2f' % b)
    for experiment in experiments:
        plt.scatter(experiment['sigma'],
                    experiment['E'], 
                    c='yellow',
                    marker='^',
                    s=300, 
                    edgecolors='white') 
   
    plt.legend(['Stocks', 'Optimal portfolio with b = %.2f' % b,'Samples portfolios'])
    plt.xlabel('σ', size=15)
    plt.ylabel('E', size=15)
    plt.title('Profitability/Risk Map', size=16)
    plt.show()

def plot_map_with_three_portfolio(all_stocks, cov_matrix, mean_vec, X, X_est, X_est_2):
    plt.grid()
    plt.scatter(data=all_stocks, x='σ', y='E', c='#6C8CD5', label='Stocks')
    plt.legend(["Stocks"])
    plt.scatter(risk_portfolio(X, cov_matrix), np.dot(mean_vec,X), color = 'red', s = 200, marker='^', edgecolors='white', label='True portfolio')
    plt.scatter(risk_portfolio(X_est, cov_matrix), np.dot(mean_vec,X_est), color ='green', s = 200, marker='^', edgecolors='white', label='Sample portfolio with estimates' )
    plt.scatter(risk_portfolio(X_est_2, cov_matrix), np.dot(mean_vec, X_est_2), color ='yellow', s = 200, marker='^', edgecolors='white', label='Sample portfolio with true mean vector')
    plt.xlabel('σ', size=15)
    plt.ylabel('E', size=15)
    plt.title('Profitability/Risk Map', size=16)
    plt.legend()
    plt.show()


def plot_samples_CVAR(all_stocks, cvar_experiments, T, N, X, cov_matrix, mean_vec, b):
    plt.grid()
    plt.scatter(data=all_stocks, x='σ', y='E', c='#6C8CD5', label='Stocks', edgecolors = 'white')
    plt.scatter(risk_portfolio(X, cov_matrix),
            np.dot(mean_vec,X), 
            c='red',
            marker='^',
            s=300, 
            edgecolors='white')
    for experiment in cvar_experiments:
        plt.scatter(experiment['sigma'],
                    experiment['E'], 
                    c='yellow',
                    marker='^',
                    s=300, 
                    edgecolors='white')

    plt.legend(['Stocks', 'Optimal portfolio with b = %.2f' % b,'Samples portfolios CVAR'])
    plt.xlabel('σ', size=15)
    plt.ylabel('E', size=15)
    plt.title('Profitability/Risk Map', size=16)



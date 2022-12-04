from functions import get_parameter_b, cvar_objective_function, cvar_optimize_portfolio, plot_shares, risk_portfolio, np, norm, plt
    
def task_3(r_matrix_gen, beta, X, stocks, X_est, df_for_graph_selected, mean_vec, cov_matrix, mean_vec_est, cov_matrix_est):
    
    b = get_parameter_b(beta)
    result = cvar_optimize_portfolio(r_matrix_gen, beta)
    T = 30
    N = 20
    UXalpha = result
    cvar_X_est = UXalpha[T:N+T]
    alpha_est  = UXalpha[-1]

    print('Alpha CVAR портфеля:' ,round(alpha_est, 4))
    print('VAR CVAR портфеля на уровне 0.95:' ,round(np.quantile(-np.dot(r_matrix_gen, cvar_X_est), 0.95), 4))
    plot_shares(X, stocks)
    plt.title('Истинные веса портфеля')
    plt.show()
    plot_shares(X_est, stocks)
    plt.title('Оценки полученных весов портфеля')
    plot_shares(cvar_X_est, stocks)
    plt.title('Оценки полученных весов портфеля CVAR') 

    print('L1 норма вектора X-Xest в 2.3 :', np.around(np.linalg.norm(X - X_est, ord=1), 3))
    print('L1 норма вектора для CVAR X-cvar_Xest:', np.around(np.linalg.norm(X - cvar_X_est, ord=1), 3))

    plt.scatter(data=df_for_graph_selected, x='σ', y='E', c='#6C8CD5', label='Stocks')
    plt.scatter(risk_portfolio(X, cov_matrix),
                np.dot(mean_vec,X), 
                c='yellow',
                marker='^',
                s=300, 
                edgecolors='black',
                label='Оптимальный портфель b = %.2f' % b)

    plt.scatter(risk_portfolio(X_est, cov_matrix_est),
                np.dot(mean_vec_est, X_est), 
                c='red',
                marker='^',
                s=300, 
                edgecolors='black',
                label='Оценка оптимального портфеля b = %.2f' % b)

    plt.scatter(risk_portfolio(cvar_X_est, cov_matrix_est),
                np.dot(mean_vec_est, cvar_X_est), 
                c='green',
                marker='^',
                s=300, 
                edgecolors='black',
                label='Оценка оптимального портфеля  CVaR' % b)
    plt.legend()
    plt.show()

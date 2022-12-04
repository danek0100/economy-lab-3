from functions import *


def task_2(mean_vec, cov_matrix, selected_stocks, beta, df_for_graph_selected,X):
    b = get_parameter_b(beta)
    T = 30
    r_matrix_gen = np.random.multivariate_normal(mean_vec, cov_matrix, T)
    mean_vec_est = np.mean(r_matrix_gen, axis=0)
    cov_matrix_est = np.cov(r_matrix_gen.T)
    
    print('Истинный вектор средних:')
    pprint.pprint(np.around(mean_vec, 3))
    print()
    print('Оценки вектора средних:')
    pprint.pprint(np.around(mean_vec_est, 3))

    X_est = optimize_portfolio(mean_vec_est, cov_matrix_est, b, bounds=((0, 1),))
    plot_shares(X_est, selected_stocks)
    plot_map_with_one_portfolio(X_est, mean_vec, cov_matrix, df_for_graph_selected, b)

    print('Истинные веса портфеля:')
    pprint.pprint(np.around(X, 3))
    print('Оценки весов портфеля:')
    pprint.pprint(np.around(X_est, 3))
    plot_map_with_two_portfolio(df_for_graph_selected, cov_matrix, mean_vec, X, X_est, mean_vec_est, cov_matrix_est)
    plot_shares(X, selected_stocks)
    plot_shares(X_est, selected_stocks)

    print('L1 норма вектора X - Xest:', np.around(np.linalg.norm(X - X_est, ord=1), 3))

    S = 40 

    experiments = []
    for i in tqdm_notebook(range(S)):
        experiment = {}
        experiment['i'] = i
        experiment['r_matrix_gen'] = np.random.multivariate_normal(mean_vec, cov_matrix, T)
        experiment['mean_vec_est'] = np.mean(experiment['r_matrix_gen'], axis=0)
        experiment['cov_matrix_est'] = np.cov(experiment['r_matrix_gen'].T)
        experiment['X_est'] =  optimize_portfolio( experiment['mean_vec_est'], 
                                                experiment['cov_matrix_est'], 
                                                b, 
                                                bounds=((0, 1),))
        experiment['L1-norm'] = np.linalg.norm(X - experiment['X_est'], ord=1)
        experiments.append(experiment)

    
    l1_norms = [experiment['L1-norm'] for experiment in experiments]
    l1_norms_mean  = np.mean(l1_norms)
    l1_norms_std  = np.std(l1_norms)
    print('Средняя L1-норма по %d экспериментам: %.3f' % (S, l1_norms_mean))
    print('Примерный 95 прц. доверительный интервал: [%.3f, %.3f]' %
        (l1_norms_mean - 2*l1_norms_std, l1_norms_mean + 2*l1_norms_std))

    
    sns.scatterplot(data=df_for_graph_selected, x='σ', y='E', c='#6C8CD5', label='Stocks')
    for experiment in experiments:
        plt.scatter(risk_portfolio(experiment['X_est'], experiment['cov_matrix_est']),
                    np.dot(experiment['mean_vec_est'], experiment['X_est']), 
                    c='red',
                    marker='^',
                    s=300, 
                    edgecolors='black')
        
    plt.scatter(risk_portfolio(X, cov_matrix),
                np.dot(mean_vec,X), 
                c='yellow',
                marker='^',
                s=300, 
                edgecolors='black',
                label='Оптимальный портфель b = %.2f' % b)
    plt.legend()
    plt.show()

    experiments_with_true_mean = []
    for i in tqdm_notebook(range(S)):
        experiment = {}
        experiment['i'] = i
        experiment['r_matrix_gen'] = np.random.multivariate_normal(mean_vec, cov_matrix, T)
        experiment['mean_vec_est'] = np.mean(experiment['r_matrix_gen'], axis=0)
        experiment['cov_matrix_est'] = np.cov(experiment['r_matrix_gen'].T)
        experiment['X_est'] =  optimize_portfolio(mean_vec,                     # используем истинный вектор средних
                                                experiment['cov_matrix_est'], 
                                                b, 
                                                bounds=((0, 1),))
        experiment['L1-norm'] = np.linalg.norm(X - experiment['X_est'], ord=1)
        experiments_with_true_mean.append(experiment)
        
    l1_norms = [experiment['L1-norm'] for experiment in experiments_with_true_mean]
    l1_norms_mean  = np.mean(l1_norms)
    l1_norms_std  = np.std(l1_norms)
    print('Средняя L1-норма по %d экспериментам: %.3f' % (S, l1_norms_mean))
    print('Примерный 95 прц. доверительный интервал: [%.3f, %.3f]' %
        (l1_norms_mean - 2*l1_norms_std, l1_norms_mean + 2*l1_norms_std))

    sns.scatterplot(data=df_for_graph_selected, x='σ', y='E', c='#6C8CD5', label='Stocks')
    for experiment in experiments_with_true_mean:
        plt.scatter(risk_portfolio(experiment['X_est'], experiment['cov_matrix_est']),
                    np.dot(experiment['mean_vec_est'], experiment['X_est']), 
                    c='red',
                    marker='*',
                    s=300, 
                    edgecolors='black')
    plt.scatter(risk_portfolio(X, cov_matrix),
                np.dot(mean_vec,X), 
                c='yellow',
                marker='^',
                s=300, 
                edgecolors='black',
                label='Оптимальный портфель b = %.2f' % b)
    plt.legend()
    plt.show()

    return X_est,r_matrix_gen, mean_vec_est, cov_matrix_est











        





        
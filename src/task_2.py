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
    plot_map_with_one_portfolio(X_est, mean_vec_est, cov_matrix_est, df_for_graph_selected, b)

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
    print('Средняя L1-норма по %d экспериментам: %.3f' % (S, l1_norms_mean))
    plot_result_experiments(df_for_graph_selected, experiments, X, cov_matrix, mean_vec, b )


    r_matrix_gen_2 = np.random.multivariate_normal(mean_vec, cov_matrix, T)
    cov_matrix_est_2 = np.cov(r_matrix_gen_2.T)
    X_est_2 = optimize_portfolio(mean_vec, cov_matrix_est, b, bounds=((0, 1),))
    
    plot_shares(X_est_2, selected_stocks)
    plot_map_with_one_portfolio(X_est_2, mean_vec, cov_matrix_est_2, df_for_graph_selected, b)
    plot_map_with_two_portfolio(df_for_graph_selected, cov_matrix, mean_vec, X, X_est_2, mean_vec, cov_matrix_est_2)

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
    print('Средняя L1-норма по %d экспериментам: %.3f' % (S, l1_norms_mean))
    plot_result_experiments(df_for_graph_selected, experiments_with_true_mean, X, cov_matrix, mean_vec, b )

    plot_map_with_three_portfolio(df_for_graph_selected, cov_matrix, mean_vec, X, X_est, mean_vec_est, cov_matrix_est,  X_est_2, mean_vec, cov_matrix_est_2)
    
    return X_est,r_matrix_gen, mean_vec_est, cov_matrix_est











        





        
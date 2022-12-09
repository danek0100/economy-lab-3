from functions import *


def task_2(mean_vec, cov_matrix, selected_stocks, beta, df_for_graph_selected,X):
    b = get_parameter_b(beta)
    ########################################################################################################################
    #2.1 and 2.2                                          
    ########################################################################################################################
    T = 30
    generated_data = np.random.multivariate_normal(mean_vec, cov_matrix, T)
    mean_vec_generated = np.mean(generated_data, axis=0)
    cov_matrix_generated = np.cov(generated_data.T)
    
    ########################################################################################################################
    #2.3                                                        
    ########################################################################################################################

    X_generated = optimize_portfolio(mean_vec_generated, cov_matrix_generated, b, bounds=((0, 1),))

    ########################################################################################################################
    #2.4                                                        
    ########################################################################################################################

    plot_shares(X_generated, selected_stocks)
    plot_map_with_one_portfolio(X_generated, mean_vec, cov_matrix, df_for_graph_selected, b)
    plot_map_with_two_portfolio(df_for_graph_selected, cov_matrix, mean_vec, X, X_generated)
    plot_shares(X, selected_stocks)
    plot_shares(X_generated, selected_stocks)

    print('L1 норма вектора X - X_est:', np.around(np.linalg.norm(X - X_generated, ord=1), 3))

    ########################################################################################################################
    #2.5                                                        
    ########################################################################################################################

    S = 40 
    experiments = []
    for i in tqdm_notebook(range(S)):
        experiment = {}
        #experiment['idx'] = i
        experiment['generated_data'] = np.random.multivariate_normal(mean_vec, cov_matrix, T)
        experiment['mean_vec'] = np.mean(experiment['generated_data'], axis=0)
        experiment['cov_matrix'] = np.cov(experiment['generated_data'].T)
        experiment['shapes'] =  optimize_portfolio( experiment['mean_vec'], experiment['cov_matrix'], b, bounds=((0, 1),))
        experiment['L1-norm'] = np.linalg.norm(X - experiment['shapes'], ord=1)
        experiment['sigma'] = risk_portfolio(experiment['shapes'], cov_matrix)
        experiment['E'] = np.dot(experiment['shapes'], mean_vec)
        experiments.append(experiment)
    
    l1_norms = [experiment['L1-norm'] for experiment in experiments]
    l1_norms_mean  = np.mean(l1_norms)
    print('Средняя L1-норма по %d экспериментам: %.3f' % (S, l1_norms_mean))
    plot_result_experiments(df_for_graph_selected, experiments, X, cov_matrix, mean_vec, b )

    ########################################################################################################################
    #2.6                                                        
    ########################################################################################################################

    generated_data_2 = np.random.multivariate_normal(mean_vec, cov_matrix, T)
    cov_matrix_generated_2 = np.cov(generated_data_2.T)
    X_generated_2 = optimize_portfolio(mean_vec, cov_matrix_generated_2, b, bounds=((0, 1),))

    print('L1 норма вектора X - X_est_2:', np.around(np.linalg.norm(X - X_generated_2, ord=1), 3))
    
    plot_shares(X_generated_2, selected_stocks)
    plot_map_with_one_portfolio(X_generated_2, mean_vec, cov_matrix_generated_2, df_for_graph_selected, b)
    plot_map_with_two_portfolio(df_for_graph_selected, cov_matrix, mean_vec, X, X_generated_2)

    experiments_with_true_mean = []
    for i in tqdm_notebook(range(S)):
        experiment = {}
        experiment['generated_data'] = np.random.multivariate_normal(mean_vec, cov_matrix, T)
        experiment['mean_vec'] = np.mean(experiment['generated_data'], axis=0)
        experiment['cov_matrix'] = np.cov(experiment['generated_data'].T)
        experiment['shapes'] =  optimize_portfolio( experiment['mean_vec'], experiment['cov_matrix'], b, bounds=((0, 1),))
        experiment['L1-norm'] = np.linalg.norm(X - experiment['shapes'], ord=1)
        experiment['sigma'] = risk_portfolio(experiment['shapes'], cov_matrix)
        experiment['E'] = np.dot(experiment['shapes'], mean_vec)
        experiments_with_true_mean.append(experiment)
        
    l1_norms = [experiment['L1-norm'] for experiment in experiments_with_true_mean]
    l1_norms_mean  = np.mean(l1_norms)
    print('Средняя L1-норма по %d экспериментам: %.3f' % (S, l1_norms_mean))
    plot_result_experiments(df_for_graph_selected, experiments_with_true_mean, X, cov_matrix, mean_vec, b )

    plot_map_with_three_portfolio(df_for_graph_selected, cov_matrix, mean_vec, X, X_generated, X_generated_2)
    
    return X_generated, generated_data, mean_vec_generated, cov_matrix_generated











        





        